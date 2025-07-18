from setuptools import setup, find_packages

setup(
    name='machine_failure_pred',
    version='0.1.0',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Add requirements, e.g.:
        'pandas',
        'numpy',
        'scikit-learn',
        'xgboost',
        'matplotlib',
        'streamlit',
    ],
)
