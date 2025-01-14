let
  pkgs = import <nixpkgs> {
    # config = {
    #   enableCuda = true;
    # };
  };
in
pkgs.mkShell {
  name = "venv-shell";
  shellHook = ''
    if [ ! -d "venv" ]; then
      python3 -m venv venv
      source venv/bin/activate
      pip install -e .
    else
      source venv/bin/activate
    fi

    export LD_LIBRARY_PATH=${
      pkgs.lib.makeLibraryPath (with pkgs; [
        pkgs.stdenv.cc.cc
        pkgs.zlib
        libGL
        glib
      ])
    }
  '';

  buildInputs = with pkgs; [
    python311Full
    python311Packages.numpy
    python311Packages.ipython
    python311Packages.magic
  ];
}
