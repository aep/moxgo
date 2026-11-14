package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/schollz/progressbar/v3"
	"github.com/spf13/cobra"

	"github.com/aep/moxgo/pkg/local"
	"github.com/aep/moxgo/pkg/registry"
)

var (
	runtimeDir        string
	runtimeFlat       bool
	runtimeNoDownload bool
	runtimeFacts      []string
	runtimeNeeds      []string
	runtimesFile      string
	bundleOut         string
	bundleDownloads   bool
	bundleCache       string
)

func init() {
	rootCmd.PersistentFlags().StringVar(&runtimeDir, "runtime-dir", "",
		"run in-process with the ONNX Runtime moxgo's runtime rules pick from the packages bundled in this directory (see 'moxgo runtime bundle')")
	rootCmd.PersistentFlags().BoolVar(&runtimeFlat, "runtime-flat", false,
		"--runtime-dir holds the packages' files directly, not one directory per package")
	rootCmd.PersistentFlags().BoolVar(&runtimeNoDownload, "no-runtime-download", false,
		"use bundled runtime packages only, never download")
	rootCmd.PersistentFlags().StringSliceVar(&runtimeFacts, "fact", nil,
		"device fact name=value for runtime rules, replacing the detected one (repeatable)")
	rootCmd.PersistentFlags().StringVar(&runtimesFile, "runtimes", "",
		"runtime rules file to use instead of moxgo's built-in ones")

	runtimeCmd.Flags().StringSliceVar(&runtimeNeeds, "needs", nil, "model capabilities to resolve for (comma-separated)")
	rootCmd.AddCommand(runtimeCmd)

	bundleCmd.Flags().StringVarP(&bundleOut, "out", "o", "", "directory to bundle into (required)")
	bundleCmd.Flags().BoolVar(&bundleDownloads, "downloads", false,
		"also bundle packages moxgo would download when needed")
	bundleCmd.Flags().StringVar(&bundleCache, "cache", "",
		"directory to keep fetched packages in between builds (default: the user cache directory)")
	_ = bundleCmd.MarkFlagRequired("out")
	runtimeCmd.AddCommand(bundleCmd)
}

func parseFacts() (registry.Facts, error) {
	f := registry.Facts{}
	for _, kv := range runtimeFacts {
		k, v, ok := strings.Cut(kv, "=")
		if !ok || k == "" {
			return nil, fmt.Errorf("--fact %q: want name=value", kv)
		}
		f[k] = v
	}
	return f, nil
}

func loadRuntimes() (*registry.Runtimes, error) {
	if runtimesFile == "" {
		return registry.DefaultRuntimes(), nil
	}
	data, err := os.ReadFile(runtimesFile)
	if err != nil {
		return nil, err
	}
	return registry.ParseRuntimes(data)
}

// progressBars shows download progress on stderr, a bar per file.
func progressBars() (func(file string, total, downloaded int64), func()) {
	var bar *progressbar.ProgressBar
	var current string
	update := func(file string, total, downloaded int64) {
		if file != current {
			if bar != nil {
				_ = bar.Finish()
				fmt.Fprintln(os.Stderr)
			}
			current = file
			bar = progressbar.DefaultBytes(total, "  "+file)
		}
		_ = bar.Set64(downloaded)
	}
	done := func() {
		if bar != nil {
			_ = bar.Finish()
			fmt.Fprintln(os.Stderr)
		}
	}
	return update, done
}

// runtimeOptions are the local.RuntimeOptions of --runtime-dir, nil
// without it.
func runtimeOptions() (*local.RuntimeOptions, error) {
	if runtimeDir == "" {
		return nil, nil
	}
	facts, err := parseFacts()
	if err != nil {
		return nil, err
	}
	rs, err := loadRuntimes()
	if err != nil {
		return nil, err
	}
	progress, _ := progressBars()
	return &local.RuntimeOptions{
		Bundled:    registry.Layout{Root: runtimeDir, Flat: runtimeFlat},
		NoDownload: runtimeNoDownload,
		Runtimes:   rs,
		Facts:      facts,
		Progress:   progress,
	}, nil
}

var runtimeCmd = &cobra.Command{
	Use:   "runtime",
	Short: "Show the ONNX Runtime moxgo's runtime rules give this device",
	Long: `Shows this device's facts, the runtime rules that match it (and a model
with the capabilities given by --needs), and the runtime they give: the ONNX
Runtime library, plugin execution providers, preloaded libraries, environment
and execution providers. With --runtime-dir, only rules whose packages are
bundled there or may be downloaded apply.`,
	Args: cobra.NoArgs,
	RunE: func(cmd *cobra.Command, args []string) error {
		rs, err := loadRuntimes()
		if err != nil {
			return err
		}
		facts := registry.DetectFacts()
		over, err := parseFacts()
		if err != nil {
			return err
		}
		for k, v := range over {
			facts[k] = v
		}
		fmt.Println("facts:")
		for _, k := range sortedKeys(facts) {
			fmt.Printf("  %s = %s\n", k, facts[k])
		}

		bundled := registry.Layout{Root: runtimeDir, Flat: runtimeFlat}
		downloads, err := registry.RuntimesDir()
		if err != nil {
			return err
		}
		where := func(pkg string) string {
			switch {
			case runtimeDir != "" && bundled.Has(rs, pkg):
				return "bundled"
			case rs.Packages[pkg].Download && !runtimeNoDownload:
				if (registry.Layout{Root: downloads}).Has(rs, pkg) {
					return "downloaded"
				}
				return "download when needed"
			case runtimeDir != "":
				return "missing"
			}
			return "bundle"
		}
		if runtimeDir != "" {
			rs = rs.Available(func(pkg string) bool { return where(pkg) != "missing" })
		}

		res := rs.Resolve(facts, runtimeNeeds)
		fmt.Println("rules applied, in order:")
		if len(res.Rules) == 0 {
			fmt.Println("  none")
		}
		for _, i := range res.Rules {
			r := rs.Rules[i]
			desc := r.Comment
			if desc == "" {
				desc = fmt.Sprintf("match %v needs %v", r.Match, r.Needs)
			}
			fmt.Printf("  %s\n", desc)
		}
		fmt.Println("runtime:")
		fmt.Printf("  core: %s\n", orNone(res.Core))
		for _, k := range sortedKeys(res.Plugins) {
			fmt.Printf("  plugin %s: %s\n", k, res.Plugins[k])
		}
		for _, p := range res.Preload {
			fmt.Printf("  preload: %s\n", p)
		}
		for _, k := range sortedKeys(res.Env) {
			fmt.Printf("  env %s=%s\n", k, res.Env[k])
		}
		if !res.HasEPs {
			fmt.Println("  eps: (any the runtime has)")
		} else if len(res.EPs) == 0 {
			fmt.Println("  eps: (CPU only)")
		}
		for _, ep := range res.EPs {
			var opts []string
			for _, k := range sortedKeys(ep.Options) {
				opts = append(opts, k+"="+ep.Options[k])
			}
			fmt.Printf("  ep %s %s\n", ep.Name, strings.Join(opts, " "))
		}
		fmt.Println("packages:")
		for _, p := range res.Packages() {
			fmt.Printf("  %s (%s)\n", p, where(p))
		}
		return nil
	},
}

var bundleCmd = &cobra.Command{
	Use:   "bundle",
	Short: "Fetch the runtime packages a build for some devices ships",
	Long: `Fetches the runtime packages of all rules that may match devices with the
facts given by --fact (e.g. --fact os=android --fact arch=arm64; facts not
given may be anything), checks them against the rules' sha256 and puts them
into --out, one directory per package or, with --runtime-flat, all files
directly (an Android app's jniLibs/arm64-v8a). Packages moxgo downloads when
a device needs them are left out unless --downloads.`,
	Args: cobra.NoArgs,
	RunE: func(cmd *cobra.Command, args []string) error {
		rs, err := loadRuntimes()
		if err != nil {
			return err
		}
		facts, err := parseFacts()
		if err != nil {
			return err
		}
		if len(facts) == 0 {
			return fmt.Errorf("give the devices to bundle for with --fact, e.g. --fact os=linux --fact arch=amd64")
		}
		pkgs := rs.Candidates(facts, bundleDownloads)
		if len(pkgs) == 0 {
			return fmt.Errorf("no runtime rule may match %v", facts)
		}
		cache := bundleCache
		if cache == "" {
			dir, err := os.UserCacheDir()
			if err != nil {
				return err
			}
			cache = filepath.Join(dir, "moxgo", "runtimes")
		}
		fmt.Printf("bundling %s\n", strings.Join(pkgs, ", "))
		progress, done := progressBars()
		err = registry.BundleRuntime(rs, pkgs, cache, registry.Layout{Root: bundleOut, Flat: runtimeFlat},
			func(p registry.Progress) { progress(p.File, p.Total, p.Downloaded) })
		done()
		if err != nil {
			return err
		}
		fmt.Printf("bundled in %s\n", bundleOut)
		return nil
	},
}

func orNone(s string) string {
	if s == "" {
		return "(none)"
	}
	return s
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
