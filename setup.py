from setuptools import setup, find_packages


setup(
    name="dtip",
    version="0.0.1",
    description="Diffusion Tensor Imaging Processing (DTIP) is a Python CLI wrapper for FSL and DTI-TK preprocessing and registration pipeline",
    long_description_content_type="text/markdown",
    author="Jitender Singh Virk",
    author_email="krivsj@gmail.com",
    url="https://github.com/VirkSaab/DTIP",
    license="unlicensed",
    packages=find_packages(exclude=["ez_setup", "tests", ".github"]),
    include_package_data=True,
    entry_points="""
        [console_scripts]
        dtip=dtip.main:cli
    """,
)
