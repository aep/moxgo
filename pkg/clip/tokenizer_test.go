package clip

import "testing"

// Ground truth generated from the OpenAI CLIP SimpleTokenizer (context_length=32);
// see /tmp/clip_ref.py. These pin the Go port to byte-exact CLIP behavior.
func TestTokenize(t *testing.T) {
	cases := []struct {
		text string
		want []int64 // leading non-pad tokens (rest must be 0 up to ContextLength)
	}{
		{"boat", []int64{49406, 4440, 49407}},
		{"buoy", []int64{49406, 609, 12437, 49407}},
		{"wind turbine", []int64{49406, 3630, 26880, 49407}},
		{"a photo of a cat", []int64{49406, 320, 1125, 539, 320, 2368, 49407}},
		{"person", []int64{49406, 2533, 49407}},
		{"the quick brown fox", []int64{49406, 518, 3712, 2866, 3240, 49407}},
		{"BOAT", []int64{49406, 4440, 49407}},              // lowercased
		{"  hello   world  ", []int64{49406, 3306, 1002, 49407}}, // whitespace-cleaned
		{"ship's mast", []int64{49406, 1158, 568, 27920, 49407}}, // contraction split
	}

	tok := Default()
	for _, c := range cases {
		got := tok.Tokenize(c.text, ContextLength)
		if len(got) != ContextLength {
			t.Fatalf("%q: len = %d, want %d", c.text, len(got), ContextLength)
		}
		for i := range got {
			var want int64
			if i < len(c.want) {
				want = c.want[i]
			}
			if got[i] != want {
				t.Errorf("%q: token[%d] = %d, want %d\n  got=%v", c.text, i, got[i], want, got[:len(c.want)+1])
				break
			}
		}
	}
}

func TestTruncation(t *testing.T) {
	tok := Default()
	long := ""
	for i := 0; i < 100; i++ {
		long += "word "
	}
	got := tok.Tokenize(long, ContextLength)
	if len(got) != ContextLength {
		t.Fatalf("len = %d, want %d", len(got), ContextLength)
	}
	if got[0] != StartOfText {
		t.Errorf("first token = %d, want SOT %d", got[0], StartOfText)
	}
	if got[ContextLength-1] != EndOfText {
		t.Errorf("last token = %d, want EOT %d (truncation must preserve EOT)", got[ContextLength-1], EndOfText)
	}
}
