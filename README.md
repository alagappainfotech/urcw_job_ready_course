---

# URCW Job Ready Course – Setup & Submission Guide (with Expected Outputs)

Welcome to the **URCW Job Ready Course** repository!
Follow these step-by-step instructions to **set up your environment**, **verify each command’s output**, and **submit your work correctly**.

---

## Training Resources

* **Python** [MIT OCW – Python Programming](https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/) *(Free)*
* **Python Basics Cheatsheet** [beginners_python_cheat_sheet_pcc_all.pdf](cheatsheets/beginners_python_cheat_sheet_pcc_all.pdf)
* **Python Documentation** [docs.python.org](https://docs.python.org/3/)
* **Git** [Pro Git Book](https://git-scm.com/book/en/v2
* **Github** [Udacity – Version Control with Git](https://www.udacity.com/course/version-control-with-git--ud123) *(Free)*
* **Github Cheatsheet** [git-cheat-sheet-education.pdf](cheatsheets/git-cheat-sheet-education.pdf)
* **Github Profile** https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-github-profile/customizing-your-profile?authuser=0

---

## 📋 Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Setup Instructions](#2-setup-instructions)

   * [Windows](#windows-setup)
   * [macOS / Linux](#macos--linux-setup)
3. [Verify Installation](#3-verify-installation)
4. [Submitting Your Work](#4-submitting-your-work)

---

## 1. System Requirements

✅ **Python 3.x** – [Download](https://www.python.org/downloads/)
✅ **Git** – [Download](https://git-scm.com/downloads)
✅ **Pip** – should come with Python
✅ **Code Editor** – [PyCharm](https://www.jetbrains.com/pycharm/) or [VS Code](https://code.visualstudio.com/)

---

## 2. Setup Instructions

### Windows Setup

**1 – Navigate to your projects folder**

```powershell
cd C:/Users/<your_administrator_id>/
```

**Expected Output:**

```
C:\Users\<your_administrator_id>>
```

---

**2 – Create PycharmProjects folder**

```powershell
mkdir PycharmProjects
```

**Expected Output:**

```
Directory: C:\Users\<your_administrator_id>

Mode                LastWriteTime         Length Name
----                -------------         ------ ----
d-----        <date>     <time>                 PycharmProjects
```

---

**3 – Go into the folder**

```powershell
cd PycharmProjects
```

**Expected Output:**

```
C:\Users\<your_administrator_id>\PycharmProjects>
```

---

**4 – Check Git version**

```powershell
git --version
```

**Expected Output:**

```
git version 2.43.0.windows.1
```

---

**5 – Clone the repository**

```powershell
git clone https://github.com/alagappainfotech/urcw_job_ready_course.git
```

**Expected Output:**

```
Cloning into 'urcw_job_ready_course'...
remote: Enumerating objects: XXX, done.
remote: Counting objects: 100% (XXX/XXX), done.
remote: Compressing objects: 100% (XXX/XXX), done.
remote: Total XXX (delta XX), reused XX (delta XX), pack-reused XX
Receiving objects: 100% (XXX/XXX), XX KiB | XX MiB/s, done.
Resolving deltas: 100% (XX/XX), done.
```

---

**6 – Enter the project folder**

```powershell
cd urcw_job_ready_course
```

**Expected Output:**

```
C:\Users\<your_administrator_id>\PycharmProjects\urcw_job_ready_course>
```

---

**7 – Switch to main branch**

```powershell
git checkout main
```

**Expected Output:**

```
Switched to branch 'main'
Your branch is up to date with 'origin/main'.
```

---

**8 – Pull the latest code**

```powershell
git pull origin main
```

**Expected Output:**

```
Already up to date.
```

*or*

```
Updating 123abc4..456def7
Fast-forward
 file1.txt | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)
```

---

**9 – Check Python**

```powershell
python --version
```

**Expected Output:**

```
Python 3.12.2
```

---

**10 – Create & activate virtual environment**

```powershell
python -m venv venv
source venv/Scripts/activate
```

**Expected Output:**

```
(venv) C:\Users\<your_administrator_id>\PycharmProjects\urcw_job_ready_course>
```

---

**11 – Install dependencies**

```powershell
pip install -r requirements.txt
```

**Expected Output:**

```
Collecting package1
...
Successfully installed package1-x.x.x package2-x.x.x
```

---

### macOS / Linux Setup

```bash
# 1 – Navigate to your projects folder
cd ~/

# 2 – Create PycharmProjects folder
mkdir PycharmProjects

# 3 – Enter the folder
cd PycharmProjects

# 4 – Check Git version
git --version
# Expected: git version 2.xx.x

# 5 – Clone repository
git clone https://github.com/alagappainfotech/urcw_job_ready_course.git

# 6 – Enter the project folder
cd urcw_job_ready_course

# 7 – Switch to main branch
git checkout main

# 8 – Pull latest changes
git pull origin main

# 9 – Check Python version
python3 --version
# Expected: Python 3.12.x

# 10 – Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 11 – Install dependencies
pip install -r requirements.txt
```

---

## 3. Verify Installation

```bash
pip --version
pip3 --version
python3 --version
```

**Expected Output Example:**

```
pip 24.0 from /path/to/python/site-packages (python 3.12)
pip 24.0 from /path/to/python/site-packages (python 3.12)
Python 3.12.2
```

---

## 4. Submitting Your Work

### Step 1 – Navigate to `python_outputs`

```bash
cd python_outputs
```

**Expected Output:**

```
.../urcw_job_ready_course/python_outputs>
```

---

### Step 2 – Create your directory

```bash
mkdir yourname_class_year_python
git status
```

**Expected Output:**

```
Untracked files:
  (use "git add <file>..." to include in what will be committed)
        python_outputs/yourname_class_year_python/
```

---

### Step 3 – Add your `.txt` output files

```bash
cd yourname_class_year_python
```
create a file for each lesson output for example: Basics_1_Primitive_Datatypes_and_Operators.txt where all outputs for the lessons are stored
```
git status
```

**Expected Output:**

```
Untracked files:
        python_outputs/yourname_class_year_python/Basics_1_Primitive_Datatypes_and_Operators.txt
```

---

### Step 4 – Pull latest code before committing

```bash
git pull origin main
```

**Expected Output:**

```
Already up to date.
```

---

### Step 5 – Stage your changes

```bash
git add Basics_1_Primitive_Datatypes_and_Operators.txt
git status
```

**Expected Output:**

```
On branch main
Your branch is up to date with 'origin/main'.

Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
	new file:   python_outputs/A_Abinaya_III_CS/Basics_1_Primitive_Datatypes_and_Operators.txt
```

---

### Step 6 – Commit your work

```bash
git commit -m "A_Abinaya_III_CS:Basics_1_Primitive_Datatypes_and_Operators.txt outputs completed"
```

**Expected Output:**

```
[main abc1234] Add output for lesson 1
 1 file changed, 1 insertion(+)
 create mode 100644 python_outputs/yourname_class_year_python/lesson1_output.txt
```

---

### Step 7 – Push to GitHub

```bash
git push origin main
```

**Expected Output:**

```
Enumerating objects: 5, done.
Counting objects: 100% (5/5), done.
Compressing objects: 100% (3/3), done.
Writing objects: 100% (3/3), 320 bytes | 320.00 KiB/s, done.
Total 3 (delta 0), reused 0 (delta 0)
To https://github.com/alagappainfotech/urcw_job_ready_course.git
   abc1234..def5678  main -> main
```

---

## 9. Contributing Guidelines

* Pull latest changes before pushing
* Write clear, commented, PEP 8-compliant code
* Include error handling
* Test before committing
* Use meaningful commit messages
* Follow the directory structure

## 10. Best Practices

* **Code Organization** → Meaningful names, consistent style
* **Error Handling** → Use try/except
* **Testing** → Unit tests for functions
* **Version Control** → Commit often, meaningful messages
* **Documentation** → Clear docstrings & updated README

---