with import <nixpkgs> {};
let
  pythonEnv = python313.withPackages (ps: [
    ps.ipython
    ps.jupyter
    ps.matplotlib
    ps.pandas
    ps.plotnine
    ps.numpy
    ps.openpyxl
    ps.polars
    ps.scipy
    ps.scikit-learn
  ]);
in mkShell {
  packages = [
    pythonEnv

    black
    git
    mypy
  ];

  shellHook = "jupyter notebook";
}