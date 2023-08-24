from setuptools import find_packages, setup
with open("README.md", "r") as fh:
    LONGDESCRIPTION = fh.read()
  

VERSION = '0.0.1' 
DESCRIPTION = 'a package that includes research in 3D rendering'

setup(
    name='nv_rd',
    packages=find_packages(include=['package_code_folder'], exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    version=VERSION,
    description=DESCRIPTION,
    long_description = LONGDESCRIPTION,
    long_description_content_type = "text/markdown",
    author='Jaadari Fadi',
    author_email="zeko12800@gmail.com",
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)



  
