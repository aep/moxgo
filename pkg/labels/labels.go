// Package labels provides class label sets for ONNX model outputs.
// Common label sets (COCO) are embedded; custom ones can be loaded from file.
package labels

import (
	"bufio"
	_ "embed"
	"fmt"
	"os"
	"strings"
)

//go:embed coco80.names
var coco80Raw string

//go:embed coco91.names
var coco91Raw string

// Labels is an ordered list of class names indexed by class ID.
type Labels []string

// Get returns the label for a class index, or "class_N" if out of range.
func (l Labels) Get(index int) string {
	if index >= 0 && index < len(l) && l[index] != "" {
		return l[index]
	}
	return "class_" + itoa(index)
}

var (
	COCO80 = parse(coco80Raw)
	COCO91 = parse(coco91Raw)
)

// ForCount returns a built-in label set matching the given class count, or nil.
func ForCount(n int) Labels {
	switch n {
	case 80:
		return COCO80
	case 91:
		return COCO91
	default:
		return nil
	}
}

// Load reads labels from a file. Supports:
//   - Plain text: one label per line
//   - CSV/TSV: uses the last non-numeric text column as label (auto-detects separator)
func Load(path string) (Labels, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close() //nolint:errcheck

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	// Peek at first line to detect format
	if !scanner.Scan() {
		return nil, fmt.Errorf("labels: empty file")
	}
	firstLine := scanner.Text()
	// Strip BOM
	firstLine = strings.TrimPrefix(firstLine, "\xef\xbb\xbf")

	sep := detectSeparator(firstLine)
	if sep == 0 {
		// Plain text format
		var lbls Labels
		lbls = append(lbls, strings.TrimSpace(firstLine))
		for scanner.Scan() {
			lbls = append(lbls, strings.TrimSpace(scanner.Text()))
		}
		return lbls, scanner.Err()
	}

	// CSV format: find label column from header
	header := strings.Split(firstLine, string(sep))
	labelCol := findLabelColumn(header)

	var lbls Labels
	for scanner.Scan() {
		line := scanner.Text()
		fields := strings.Split(line, string(sep))
		if labelCol < len(fields) {
			lbls = append(lbls, strings.TrimSpace(fields[labelCol]))
		} else {
			lbls = append(lbls, "")
		}
	}
	return lbls, scanner.Err()
}

func detectSeparator(line string) byte {
	if strings.Contains(line, ";") {
		return ';'
	}
	if strings.Contains(line, "\t") {
		return '\t'
	}
	if strings.Count(line, ",") >= 2 {
		return ','
	}
	return 0
}

func findLabelColumn(header []string) int {
	// Prefer columns named "com_name", "label", "name", "class_name"
	preferred := []string{"com_name", "label", "name", "class_name", "display_name"}
	for _, want := range preferred {
		for i, h := range header {
			if strings.EqualFold(strings.TrimSpace(h), want) {
				return i
			}
		}
	}
	// Fall back to last column
	return len(header) - 1
}

func parse(raw string) Labels {
	lines := strings.Split(strings.TrimRight(raw, "\n"), "\n")
	labels := make(Labels, len(lines))
	for i, line := range lines {
		labels[i] = strings.TrimSpace(line)
	}
	return labels
}

func itoa(n int) string {
	if n < 0 {
		return "-" + uitoa(uint(-n))
	}
	return uitoa(uint(n))
}

func uitoa(n uint) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
