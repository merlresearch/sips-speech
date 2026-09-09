# Copyright (c) 2026 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent


def read_text(path: str) -> str:
    file_path = ROOT / path
    if not file_path.exists():
        return ""
    return file_path.read_text(encoding="utf-8")


def read_requirements(path: str = "requirements.txt") -> list[str]:
    file_path = ROOT / path
    if not file_path.exists():
        return []

    requirements = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # The import module is named "yaml", but the PyPI package is PyYAML.
        if line == "yaml":
            line = "PyYAML"

        requirements.append(line)

    return requirements


install_requires = sorted(set(read_requirements()))

extras_require = {
    "dev": [
        "black==24.4.2",
        "flake8==7.1.0",
        "isort==5.13.2",
        "pre-commit==3.7.1",
        "pytest",
    ],
}


setup(
    name="sips_speech",
    version="0.1.0",
    description="SIPS: Stochastic Interpolant Prior for Speech",
    long_description=read_text("README.md"),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["sips_speech", "sips_speech.*"]),
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "sips-train=sips_speech.train:main",
            "sips-infer=sips_speech.infer:main",
        ],
    },
)
