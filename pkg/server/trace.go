package server

import (
	"context"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

// Inference is traced through the OpenTelemetry API: a program embedding
// moxgo that sets a tracer provider gets a "moxgo.predict" span per traced
// prediction, with the stages as children, under the span of the context it
// passes to Instance.PredictContext or Instance.Detect. A prediction whose
// context carries no recording span (no provider, or a trace not sampled)
// starts none: spans cost allocations even when nothing records them.

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

// noSpan is the span of a prediction not traced.
var noSpan = trace.SpanFromContext(context.Background())

// startSpan starts a span under ctx's, when that one records.
func startSpan(ctx context.Context, name string, opts ...trace.SpanStartOption) (context.Context, trace.Span) {
	if !trace.SpanFromContext(ctx).IsRecording() {
		return ctx, noSpan
	}
	return tracer.Start(ctx, name, opts...)
}

// startPredict starts the span of a prediction, when ctx's records.
func (inst *Instance) startPredict(ctx context.Context) (context.Context, trace.Span) {
	if !trace.SpanFromContext(ctx).IsRecording() {
		return ctx, noSpan
	}
	return tracer.Start(ctx, "moxgo.predict", inst.spanAttrs())
}

func (inst *Instance) spanAttrs() trace.SpanStartOption {
	return trace.WithAttributes(
		attrModel.String(inst.Name),
		attrEP.String(inst.ActiveEP),
		attrThreads.Int(inst.Threads),
	)
}
