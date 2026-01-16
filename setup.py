from setuptools import find_packages, setup


def get_requirements(file_path):
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")

    return requirements


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


setup(
    name='predicting_road_accident_risk',
    version='0.0.1',
    author='Christopher Christain G',
    author_email='christopherchristain.02@gmail.com',
    description='End-to-end machine learning project for predicting road accident risk',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.9"
)
