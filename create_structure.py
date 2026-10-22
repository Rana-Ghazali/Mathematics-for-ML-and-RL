
from pathlib import Path

root = Path(".")

structure = {
    "01-Linear-Algebra": [
        "01-Vectors-and-Vector-Operations",
        "02-Euclidean-and-Manhattan-Distance",
        "03-Dot-and-Cross-Product",
        "04-Matrix-Operations",
        "05-Systems-of-Linear-Equations",
        "06-Orthogonality-and-Projection",
        "07-Eigenvalues-and-Eigenvectors",
        "08-Singular-Value-Decomposition",
        "09-Cosine-and-Jaccard-Similarity",
        "10-Principal-Component-Analysis",
    ],
    "02-Probability": [],
    "03-Statistics": [],
    "04-Calculus": [],
}

for section, lessons in structure.items():
    section_path = root / section
    section_path.mkdir(parents=True, exist_ok=True)

    section_readme = section_path / "README.md"
    if not section_readme.exists():
        section_readme.write_text(
            f"# {section.replace('-', ' ').replace('01 ', '').replace('02 ', '').replace('03 ', '').replace('04 ', '')}\n\n"
            "Mathematical foundations for Machine Learning and Reinforcement Learning.\n",
            encoding="utf-8",
        )

    for lesson in lessons:
        lesson_path = section_path / lesson
        lesson_path.mkdir(parents=True, exist_ok=True)

        readme = lesson_path / "README.md"
        if not readme.exists():
            title = lesson[3:].replace("-", " ")
            readme.write_text(
                f"# {title}\n\n"
                "## 1. Introduction\n\n"
                "## 2. Mathematical Foundations\n\n"
                "## 3. Worked Examples\n\n"
                "## 4. Python Implementation\n\n"
                "## 5. Applications in Machine Learning\n\n"
                "## 6. Applications in Reinforcement Learning\n\n"
                "## 7. Exercises\n",
                encoding="utf-8",
            )

print("Repository structure created successfully!")
