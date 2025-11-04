# c1lgkt-jax
Lagrangian Gyrokinetic tracer code with C1 interpolation on fields, now in JAX!

## Folder Structure
- `docs` - Contains documentation for this package; currently nonexistent
- `src/c1lgkt/jax` - Contains all of the packagable code for the particle pushing in JAX
- `bin` - Directory which contains the "working codes". Note that these are implcitly run with respect to the workspace root, i.e. the folder containing this README

## Running the Codes
In the `bin` folder, I've set up the scripts so they're implicitly run from the root directory of this project. Here's an example of my .vscode/settings.json file
```json
{
    "jupyter.notebookFileRoot": "${workspaceFolder}",
    "python.analysis.extraPaths": ["./src"],
    "python.autoComplete.extraPaths": [
        "./src"
    ],
    "python-envs.defaultEnvManager": "ms-python.python:conda",
    "python-envs.defaultPackageManager": "ms-python.python:conda",
    "python-envs.pythonProjects": [],
    "python.envFile": "${workspaceFolder}/.env",
    "terminal.integrated.env.windows": {
        "PYTHONPATH": "${env:PYTHONPATH};${workspaceFolder}/src"
    }
}
```