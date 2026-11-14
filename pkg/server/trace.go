package server

import (
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

// Inference is traced through the OpenTelemetry API: with no tracer provider
// set, as in moxgo's own commands, spans cost nothing; a program embedding
// moxgo that sets one gets a "moxgo.predict" span per traced prediction, with
// the stages as children, under the span of the context it passes to
// Instance.PredictContext.

var tracer = otel.Tracer("github.com/aep/moxgo/pkg/server")

// Span attributes.
const (
	attrModel     = attribute.Key("moxgo.model")
	attrEP        = attribute.Key("moxgo.execution_provider")
	attrThreads   = attribute.Key("moxgo.threads")
	attrInputSize = attribute.Key("moxgo.input.bytes")
	attrOutputs   = attribute.Key("moxgo.outputs")
	attrWarmup    = attribute.Key("moxgo.warmup")
)

func (inst *Instance) spanAttrs() trace.SpanStartOption {
	return trace.WithAttributes(
		attrModel.String(inst.Name),
		attrEP.String(inst.ActiveEP),
		attrThreads.Int(inst.Threads),
	)
}
