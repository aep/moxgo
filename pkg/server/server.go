package server

import (
	"context"
	"fmt"
	"net/http"
	"strings"

	"connectrpc.com/connect"

	"github.com/aep/moxgo/pkg/onnx"
	gomaxv1 "github.com/aep/moxgo/pkg/proto/gomax/v1"
	"github.com/aep/moxgo/pkg/proto/gomax/v1/gomaxv1connect"
	"github.com/aep/moxgo/pkg/version"
)

type PullFunc func(name string, progress func(file string, total, downloaded int64)) (*ModelConfig, error)

type Server struct {
	rt   *onnx.Runtime
	mgr  *InstanceManager
	pull PullFunc
}

func New(rt *onnx.Runtime, reg *Registry, pull PullFunc) *Server {
	return &Server{
		rt:   rt,
		mgr:  NewInstanceManager(rt, reg),
		pull: pull,
	}
}

func (s *Server) Handler() (string, http.Handler) {
	return gomaxv1connect.NewInferenceServiceHandler(s)
}

func (s *Server) Shutdown() {
	s.mgr.Shutdown()
}

func (s *Server) Info(_ context.Context, _ *connect.Request[gomaxv1.InfoRequest]) (*connect.Response[gomaxv1.InfoResponse], error) {
	return connect.NewResponse(&gomaxv1.InfoResponse{Version: version.Version}), nil
}

func (s *Server) ListProviders(_ context.Context, _ *connect.Request[gomaxv1.ListProvidersRequest]) (*connect.Response[gomaxv1.ListProvidersResponse], error) {
	providers, err := s.rt.GetAvailableProviders()
	if err != nil {
		return nil, connect.NewError(connect.CodeInternal, err)
	}
	short := make([]string, len(providers))
	for i, p := range providers {
		short[i] = strings.TrimSuffix(p, "ExecutionProvider")
	}
	return connect.NewResponse(&gomaxv1.ListProvidersResponse{Providers: short}), nil
}

func (s *Server) List(_ context.Context, _ *connect.Request[gomaxv1.ListRequest]) (*connect.Response[gomaxv1.ListResponse], error) {
	models := s.mgr.registry.List()
	resp := &gomaxv1.ListResponse{Models: make([]*gomaxv1.ModelInfo, len(models))}
	for i, cfg := range models {
		mi := &gomaxv1.ModelInfo{
			Name: cfg.Name,
			Path: cfg.Path,
		}
		for _, inp := range cfg.InputList {
			mi.Inputs = append(mi.Inputs, &gomaxv1.ModelInput{
				Name:   inp.Name,
				Type:   inp.Type,
				Params: inp.ParamsMap(),
			})
		}
		resp.Models[i] = mi
	}
	return connect.NewResponse(resp), nil
}

func (s *Server) Ps(_ context.Context, _ *connect.Request[gomaxv1.PsRequest]) (*connect.Response[gomaxv1.PsResponse], error) {
	instances := s.mgr.List()
	resp := &gomaxv1.PsResponse{Models: make([]*gomaxv1.RunningModel, len(instances))}
	for i, inst := range instances {
		resp.Models[i] = &gomaxv1.RunningModel{
			Name:    inst.Name,
			Ep:      inst.ActiveEP,
			Threads: int32(inst.Threads),
		}
	}
	return connect.NewResponse(resp), nil
}

func (s *Server) Pull(_ context.Context, req *connect.Request[gomaxv1.PullRequest], stream *connect.ServerStream[gomaxv1.PullEvent]) error {
	if s.pull == nil {
		return connect.NewError(connect.CodeUnimplemented, fmt.Errorf("pull not configured"))
	}
	cfg, err := s.pull(req.Msg.Model, func(file string, total, downloaded int64) {
		_ = stream.Send(&gomaxv1.PullEvent{
			File:       file,
			Total:      total,
			Downloaded: downloaded,
		})
	})
	if err != nil {
		return connect.NewError(connect.CodeInternal, err)
	}
	if err := s.mgr.registry.Add(cfg); err != nil {
		return connect.NewError(connect.CodeInternal, err)
	}
	return stream.Send(&gomaxv1.PullEvent{Done: true})
}

func (s *Server) Run(_ context.Context, req *connect.Request[gomaxv1.RunRequest]) (*connect.Response[gomaxv1.RunResponse], error) {
	inst, err := s.mgr.Run(req.Msg.Model, req.Msg.Ep)
	if err != nil {
		if strings.Contains(err.Error(), "not found in registry") {
			return nil, connect.NewError(connect.CodeNotFound, err)
		}
		return nil, connect.NewError(connect.CodeInternal, err)
	}
	return connect.NewResponse(&gomaxv1.RunResponse{
		Model: inst.Name,
		Ep:    inst.ActiveEP,
	}), nil
}

func (s *Server) Rm(_ context.Context, req *connect.Request[gomaxv1.RmRequest]) (*connect.Response[gomaxv1.RmResponse], error) {
	if err := s.mgr.Rm(req.Msg.Model); err != nil {
		return nil, connect.NewError(connect.CodeNotFound, err)
	}
	return connect.NewResponse(&gomaxv1.RmResponse{}), nil
}

func (s *Server) Predict(_ context.Context, req *connect.Request[gomaxv1.PredictRequest]) (*connect.Response[gomaxv1.PredictResponse], error) {
	inst, err := s.mgr.GetOrRun(req.Msg.Model)
	if err != nil {
		return nil, connect.NewError(connect.CodeNotFound, err)
	}
	resp, err := inst.Predict(req.Msg.Inputs)
	if err != nil {
		return nil, connect.NewError(connect.CodeInternal, err)
	}
	return connect.NewResponse(resp), nil
}

func (s *Server) PredictStream(_ context.Context, req *connect.Request[gomaxv1.PredictStreamRequest], stream *connect.ServerStream[gomaxv1.PredictStreamResponse]) error {
	inst, err := s.mgr.GetOrRun(req.Msg.Model)
	if err != nil {
		return connect.NewError(connect.CodeNotFound, err)
	}
	return inst.PredictChunked(req.Msg.Inputs, func(resp *gomaxv1.PredictStreamResponse) error {
		return stream.Send(resp)
	})
}

func (s *Server) Manager() *InstanceManager {
	return s.mgr
}

// Preload loads a model by name so it's ready for inference.
func (s *Server) Preload(name string) error {
	_, err := s.mgr.Run(name, "")
	if err != nil {
		return fmt.Errorf("preload %s: %w", name, err)
	}
	return nil
}
