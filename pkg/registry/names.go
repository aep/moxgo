package registry

import (
	"fmt"
	"regexp"
	"strings"
)

// Models have variants, named like ollama's: "yolo11n" is the default
// variant, the same as "yolo11n:latest", and "yolo11n:int8" a quantized
// one. The registry lists a model's variants with it (IndexEntry.Variants)
// and serves the default's manifest at /v1/<model>.json, another variant's
// at /v1/<model>/<tag>.json. Locally a variant lives in models/<model>@<tag>
// (":" is not allowed in Windows file names).

// DefaultTag names a model's default variant.
const DefaultTag = "latest"

var (
	modelNameRe = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9._-]*$`)
	tagRe       = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9._-]*$`)
)

// SplitName splits "model:tag" into the model and its tag; the default
// variant has tag "". It rejects names that are no model name.
func SplitName(name string) (model, tag string, err error) {
	model, tag, hasTag := strings.Cut(name, ":")
	if hasTag && tag == "" {
		return "", "", fmt.Errorf("bad model name %q: empty variant", name)
	}
	if tag == DefaultTag {
		tag = ""
	}
	if !modelNameRe.MatchString(model) {
		return "", "", fmt.Errorf("bad model name %q", name)
	}
	if tag != "" && !tagRe.MatchString(tag) {
		return "", "", fmt.Errorf("bad variant %q in model name %q", tag, name)
	}
	return model, tag, nil
}

// CanonicalName is name without a default ":latest" tag.
func CanonicalName(name string) string {
	return strings.TrimSuffix(name, ":"+DefaultTag)
}

// manifestPath is the registry path of a model variant's manifest.
func manifestPath(model, tag string) string {
	if tag == "" {
		return "/v1/" + model + ".json"
	}
	return "/v1/" + model + "/" + tag + ".json"
}

// dirName is the local directory name of a model variant.
func dirName(model, tag string) string {
	if tag == "" {
		return model
	}
	return model + "@" + tag
}

// nameOfDir is the model name of a local directory.
func nameOfDir(dir string) string {
	if model, tag, ok := strings.Cut(dir, "@"); ok {
		return model + ":" + tag
	}
	return dir
}
