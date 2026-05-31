{
  description = "moxgo cross-compile toolchain (linux/windows/darwin/android) via nix";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; config.allowUnfree = true; config.android_sdk.accept_license = true; };
        ndk = pkgs.androidenv.androidPkgs.ndk-bundle or pkgs.androidndk;
        macos-sdk = pkgs.stdenvNoCC.mkDerivation {
          name = "MacOSX13.3.sdk";
          src = pkgs.fetchurl {
            url = "https://github.com/joseluisq/macosx-sdks/releases/download/13.3/MacOSX13.3.sdk.tar.xz";
            sha256 = "sha256-UY416uYDmz9k6AJfRSXBxDeGzFzzlFnWCYUvrwkeNL4=";
          };
          unpackPhase = "tar -xJf $src";
          installPhase = "mv MacOSX13.3.sdk $out";
        };
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.zig
            pkgs.llvm
            pkgs.go
            pkgs.gnumake
            pkgs.pkg-config
            pkgs.govulncheck
            pkgs.golangci-lint
            ndk
          ];

          shellHook = ''
            export ANDROID_NDK="$(find ${ndk}/libexec/android-sdk/ndk -maxdepth 1 -mindepth 1 -type d | head -1)"
            export MACOS_SDK="${macos-sdk}"
            echo "moxgo cross-compile shell ready. Run: make cross"
          '';
        };
      });
}
