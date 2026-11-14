package cmd

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/aep/moxgo/pkg/onnx"
	"github.com/aep/moxgo/pkg/sam3"
	"github.com/spf13/cobra"
)

var (
	sam3Dir       string
	sam3Prompts   string
	sam3Threshold float32
	sam3EP        string
	sam3MaskDir   string
)

func init() {
	rootCmd.AddCommand(sam3Cmd)
	sam3Cmd.Flags().StringVar(&sam3Dir, "dir", "/shared/sam3", "directory with the SAM3 onnx models + config.yaml")
	sam3Cmd.Flags().StringVar(&sam3Prompts, "prompt", "", "comma-separated text prompts (e.g. \"boat,buoy,wind turbine\")")
	sam3Cmd.Flags().Float32Var(&sam3Threshold, "threshold", 0.5, "score threshold")
	sam3Cmd.Flags().StringVar(&sam3EP, "ep", "", "execution provider override (CUDA, CPU; empty=auto)")
	sam3Cmd.Flags().StringVar(&sam3MaskDir, "masks", "", "if set, write per-detection mask PNGs to this directory")
}

var sam3Cmd = &cobra.Command{
	Use:   "sam3 <image>",
	Short: "Run the SAM 3 open-vocabulary text→box+mask pipeline on one image",
	Args:  cobra.ExactArgs(1),
	RunE: func(_ *cobra.Command, args []string) error {
		prompts := splitPrompts(sam3Prompts)
		if len(prompts) == 0 {
			return fmt.Errorf("at least one --prompt is required")
		}

		var rt *onnx.Runtime
		var err error
		if libPath != "" {
			rt, err = onnx.LoadFrom(libPath)
		} else {
			rt, err = onnx.Load()
		}
		if err != nil {
			return fmt.Errorf("load ONNX Runtime: %w", err)
		}
		defer rt.Close()

		img, err := loadImage(args[0])
		if err != nil {
			return err
		}

		t0 := time.Now()
		model, err := sam3.Open(rt, sam3Dir, sam3.Options{EP: sam3EP})
		if err != nil {
			return err
		}
		defer model.Close()
		fmt.Printf("loaded SAM3 (ep=%s) in %s\n", model.ActiveEP, time.Since(t0).Round(time.Millisecond))

		t1 := time.Now()
		dets, err := model.Detect(img, prompts, sam3Threshold)
		if err != nil {
			return err
		}
		fmt.Printf("detected %d objects in %s (prompts: %s)\n\n", len(dets), time.Since(t1).Round(time.Millisecond), strings.Join(prompts, ", "))

		for i, d := range dets {
			cover := ""
			if d.Mask != nil {
				cover = fmt.Sprintf("  mask=%dx%d (%d px)", d.Mask.W, d.Mask.H, maskArea(d.Mask))
			}
			fmt.Printf("  [%2d] %-14s score=%.3f  box=[%.1f %.1f %.1f %.1f]%s\n",
				i, d.Label, d.Score, d.Box[0], d.Box[1], d.Box[2], d.Box[3], cover)
		}

		if sam3MaskDir != "" {
			if err := os.MkdirAll(sam3MaskDir, 0o755); err != nil {
				return err
			}
			for i, d := range dets {
				if d.Mask == nil {
					continue
				}
				p := filepath.Join(sam3MaskDir, fmt.Sprintf("mask_%02d_%s.png", i, sanitize(d.Label)))
				if err := writeMaskPNG(p, d.Mask); err != nil {
					return err
				}
			}
			fmt.Printf("\nwrote masks to %s\n", sam3MaskDir)
		}
		return nil
	},
}

func splitPrompts(s string) []string {
	var out []string
	for _, p := range strings.Split(s, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

func loadImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open image: %w", err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}
	return img, nil
}

func maskArea(m *sam3.Mask) int {
	n := 0
	for _, b := range m.Bits {
		if b {
			n++
		}
	}
	return n
}

func writeMaskPNG(path string, m *sam3.Mask) error {
	img := image.NewGray(image.Rect(0, 0, m.W, m.H))
	for y := 0; y < m.H; y++ {
		for x := 0; x < m.W; x++ {
			if m.Bits[y*m.W+x] {
				img.SetGray(x, y, color.Gray{Y: 255})
			}
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}

func sanitize(s string) string {
	return strings.ReplaceAll(strings.TrimSpace(s), " ", "_")
}
