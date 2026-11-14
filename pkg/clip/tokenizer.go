// Package clip implements the OpenAI CLIP byte-pair-encoding tokenizer, a Go
// port of CLIP's SimpleTokenizer. It turns a text prompt into the int64 token
// sequence expected by CLIP-style text encoders (e.g. the SAM 3 language
// encoder, which takes tokens [1, 32]).
package clip

import (
	"bufio"
	"bytes"
	"compress/gzip"
	_ "embed"
	"html"
	"regexp"
	"strings"
	"sync"
)

// Special tokens and the default context length used by the SAM 3 language
// encoder (CLIP defaults to 77; SAM 3 / YOLO-World use 32).
const (
	StartOfText   = 49406
	EndOfText     = 49407
	ContextLength = 32
	numMergesUsed = 49152 - 256 - 2 // 48894 merge rules retained from the vocab file
)

//go:embed bpe_simple_vocab_16e6.txt.gz
var bpeVocabGz []byte

// splitPat mirrors CLIP's tokenization regex. RE2 supports \p{L}/\p{N}, and
// Go's leftmost-first alternation matches Python's re/regex ordering, so the
// contraction rules ('s, 't, …) take precedence over the letter run.
var splitPat = regexp.MustCompile(`(?i)<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+`)

var wsPat = regexp.MustCompile(`\s+`)

// whitespaceClean collapses runs of whitespace to a single space and trims,
// matching CLIP's whitespace_clean.
func whitespaceClean(s string) string {
	return strings.TrimSpace(wsPat.ReplaceAllString(s, " "))
}

// Tokenizer is a reusable CLIP BPE tokenizer. Build once with New; Tokenize is
// safe for concurrent use except for its internal merge cache, which is guarded.
type Tokenizer struct {
	byteEncoder map[byte]rune     // byte -> printable rune (CLIP bytes_to_unicode)
	encoder     map[string]int    // BPE token string -> id
	bpeRanks    map[[2]string]int // ordered merge rule -> rank

	mu    sync.Mutex
	cache map[string]string
}

var (
	defaultOnce sync.Once
	defaultTok  *Tokenizer
)

// Default returns a lazily-built shared tokenizer.
func Default() *Tokenizer {
	defaultOnce.Do(func() { defaultTok = New() })
	return defaultTok
}

// New builds a tokenizer from the embedded CLIP vocabulary.
func New() *Tokenizer {
	byteEncoder, byteVals := bytesToUnicode()

	merges := loadMerges()

	// vocab = byteVals + byteVals+"</w>" + merged pairs + specials
	vocab := make([]string, 0, len(byteVals)*2+len(merges)+2)
	for _, r := range byteVals {
		vocab = append(vocab, string(r))
	}
	for _, r := range byteVals {
		vocab = append(vocab, string(r)+"</w>")
	}
	for _, m := range merges {
		vocab = append(vocab, m[0]+m[1])
	}
	vocab = append(vocab, "<|startoftext|>", "<|endoftext|>")

	encoder := make(map[string]int, len(vocab))
	for i, v := range vocab {
		encoder[v] = i
	}
	bpeRanks := make(map[[2]string]int, len(merges))
	for i, m := range merges {
		bpeRanks[m] = i
	}

	return &Tokenizer{
		byteEncoder: byteEncoder,
		encoder:     encoder,
		bpeRanks:    bpeRanks,
		cache: map[string]string{
			"<|startoftext|>": "<|startoftext|>",
			"<|endoftext|>":   "<|endoftext|>",
		},
	}
}

// Tokenize encodes text into a fixed-length context window: [SOT] tokens [EOT]
// padded with 0, truncated to ctx (keeping EOT as the final token). Pass
// ctx <= 0 to use ContextLength.
func (t *Tokenizer) Tokenize(text string, ctx int) []int64 {
	if ctx <= 0 {
		ctx = ContextLength
	}
	ids := t.encode(text)

	out := make([]int64, ctx)
	out[0] = StartOfText
	n := 1
	for _, id := range ids {
		if n >= ctx-1 {
			break // leave room for EOT
		}
		out[n] = int64(id)
		n++
	}
	out[n] = EndOfText
	return out
}

func (t *Tokenizer) encode(text string) []int {
	text = strings.ToLower(whitespaceClean(html.UnescapeString(html.UnescapeString(text))))

	var ids []int
	for _, tok := range splitPat.FindAllString(text, -1) {
		// Map the token's UTF-8 bytes through the byte encoder into a
		// printable-rune string, then BPE-merge it.
		var sb strings.Builder
		for i := 0; i < len(tok); i++ {
			sb.WriteRune(t.byteEncoder[tok[i]])
		}
		for _, piece := range strings.Split(t.bpe(sb.String()), " ") {
			if id, ok := t.encoder[piece]; ok {
				ids = append(ids, id)
			}
		}
	}
	return ids
}

// bpe applies the merge rules to a single byte-encoded token.
func (t *Tokenizer) bpe(token string) string {
	t.mu.Lock()
	if c, ok := t.cache[token]; ok {
		t.mu.Unlock()
		return c
	}
	t.mu.Unlock()

	runes := []rune(token)
	if len(runes) == 0 {
		return token
	}
	// word = runes, with "</w>" appended to the last symbol.
	word := make([]string, len(runes))
	for i, r := range runes {
		word[i] = string(r)
	}
	word[len(word)-1] += "</w>"

	if len(word) == 1 {
		res := word[0]
		t.store(token, res)
		return res
	}

	for {
		// Find the adjacent pair with the lowest merge rank.
		bestRank := int(^uint(0) >> 1)
		bestIdx := -1
		for i := 0; i < len(word)-1; i++ {
			if r, ok := t.bpeRanks[[2]string{word[i], word[i+1]}]; ok && r < bestRank {
				bestRank = r
				bestIdx = i
			}
		}
		if bestIdx < 0 {
			break
		}
		first, second := word[bestIdx], word[bestIdx+1]

		merged := make([]string, 0, len(word))
		i := 0
		for i < len(word) {
			if i < len(word)-1 && word[i] == first && word[i+1] == second {
				merged = append(merged, first+second)
				i += 2
			} else {
				merged = append(merged, word[i])
				i++
			}
		}
		word = merged
		if len(word) == 1 {
			break
		}
	}

	res := strings.Join(word, " ")
	t.store(token, res)
	return res
}

func (t *Tokenizer) store(k, v string) {
	t.mu.Lock()
	t.cache[k] = v
	t.mu.Unlock()
}

// bytesToUnicode reproduces CLIP's reversible byte<->unicode map. It returns the
// byte->rune map and the ordered list of runes (the first 256 vocab entries).
func bytesToUnicode() (map[byte]rune, []rune) {
	var bs []int
	add := func(lo, hi rune) {
		for c := lo; c <= hi; c++ {
			bs = append(bs, int(c))
		}
	}
	add('!', '~')
	add('¡', '¬')
	add('®', 'ÿ')

	cs := make([]int, len(bs))
	copy(cs, bs)

	inBs := make(map[int]bool, len(bs))
	for _, b := range bs {
		inBs[b] = true
	}
	n := 0
	for b := 0; b < 256; b++ {
		if !inBs[b] {
			bs = append(bs, b)
			cs = append(cs, 256+n)
			n++
		}
	}

	enc := make(map[byte]rune, 256)
	vals := make([]rune, len(bs))
	for i := range bs {
		enc[byte(bs[i])] = rune(cs[i])
		vals[i] = rune(cs[i])
	}
	return enc, vals
}

// loadMerges decompresses the embedded vocab and returns the retained merge
// rules (skipping the version header line).
func loadMerges() [][2]string {
	gz, err := gzip.NewReader(bytes.NewReader(bpeVocabGz))
	if err != nil {
		panic("clip: bad embedded vocab: " + err.Error())
	}
	defer gz.Close()

	merges := make([][2]string, 0, numMergesUsed)
	sc := bufio.NewScanner(gz)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		if line == 1 {
			continue // version header
		}
		if len(merges) >= numMergesUsed {
			break
		}
		parts := strings.Fields(sc.Text())
		if len(parts) != 2 {
			continue
		}
		merges = append(merges, [2]string{parts[0], parts[1]})
	}
	if err := sc.Err(); err != nil {
		panic("clip: scan vocab: " + err.Error())
	}
	return merges
}
