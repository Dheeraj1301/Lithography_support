import os

project_structure = {
    "config": ["config.yaml"],
    "data/raw": [],
    "data/processed": [],
    "data": ["synthetic_generator.py"],
    "notebooks": ["01_eda.ipynb", "02_modeling.ipynb"],
    "src/data": ["loader.py"],
    "src/features": ["build_features.py"],
    "src/models": ["lstm_model.py", "rl_agent.py", "trainer.py"],
    "src/envs": ["litho_env.py"],
    "src/controller": ["adaptive_controller.py", "api_integrator.py"],
    "src/utils": ["helpers.py"],
    "dashboard": ["app.py", "visualizations.py"],
    "tests": ["test_models.py", "test_env.py"],
    "scripts": ["run_training.py", "deploy_dashboard.sh"],
    "": ["requirements.txt", "README.md", ".gitignore"]
}

def create_project_structure(base_path="adaptive_lithography_ai"):
    for folder, files in project_structure.items():
        dir_path = os.path.join(base_path, folder)
        os.makedirs(dir_path, exist_ok=True)
        for file in files:
            file_path = os.path.join(dir_path, file)
            with open(file_path, "w") as f:
                pass
            print(f"📄 Created: {file_path}")
    print("\n✅ Project structure initialized.")

if __name__ == "__main__":
    create_project_structure()
