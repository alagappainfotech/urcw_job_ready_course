# Version Control

## What Is Version Control

> Version control is a system that records changes to a file or set of files over time so that you can recall specific versions later.

There are two main types of version control system models:
* **the centralized model** - all users connect to a central, master repository
* **the distributed model** - each user has the entire repository on their computer

Further reading:
- [What is version control: centralized vs. DVCS](http://blogs.atlassian.com/2012/02/version-control-centralized-dvcs/)

## Git and Version Control Terminology

### Version Control System (VCS) or Source Code Manager (SCM):

A VCS allows you to:
- revert files back to a previous state,
- revert the entire project back to a previous state,
- review changes made over time, see who last modified something that might be causing
a problem, who introduced an issue and when, and more.

Git is a **VCS**.

### Commit (snapshot):

Git thinks of its data like a set of snapshots of a mini filesystem. Every time you commit (save the state of your project in Git), it basically takes a picture of what all your files look like at that moment and stores a reference to that snapshot. You can think of it as a save point in a game - it saves your project's files and any information about them.

Everything you do in Git is to help you make commits, so a commit is the fundamental unit in Git.

### Repository (repo):

A repository is a directory which contains your project work, as well as a few files (hidden by default on Mac OS X) which are used to communicate with Git. Repositories can exist either locally on your computer or as a remote copy on another computer. A repository is made up of commits.

### Working Directory

The Working Directory is the files that you see in your computer's file system. When you open your project files up on a code editor, you're working with files in the Working Directory.

This is in contrast to the files that have been saved (in commits!) in the repository.

When working with Git, the Working Directory is also different from the command line's concept of the *current working directory* which is the directory that your shell is "looking at" right now.

### Checkout: 

When content in the repository has been copied to the Working Directory. It is possible to checkout many things from a repository; a file, a commit, a branch, etc.

### Staging Area or Staging Index or Index

A file in the Git directory that stores information about what will go into your next commit. You can think of the **staging area** as a prep table where Git will take the next commit. Files on the Staging Index are poised to be added to the repository.


### SHA

A **SHA** is basically an ID number for each commit. Here's what a commit's SHA might look like: `e2adf8ae3e2e4ed40add75cc44cf9d0a869afeb6`.

It is a 40-character string composed of characters (0–9 and a–f) and calculated based on the contents of a file or directory structure in Git. "SHA" is shorthand for "Secure Hash Algorithm".

### Branch

A branch is when a new line of development is created that diverges from the main line of development. This alternative line of development can continue without altering the main line.

Going back to [the example of save point in a game](#commit-snapshot), you can think of a branch as where you make a save point in your game and then decide to try out a risky move in the game. If the risky move doesn't pan out, then you can just go back to the save point. The key thing that makes branches incredibly powerful is that you can make save points on one branch, and then switch to a different branch and make save points there, too.

## Create a Git Repo

### Create A Repo From Scratch

Use the git init command to create a new, empty repository in the current directory.
```
$ git init
```
Running this command creates a hidden `.git` directory. This `.git` directory is the brain/storage center for the repository. It holds all of the configuration files and directories and is where all of the commits are stored.

**Helpful Links**:
* [Initializing a Repository in an Existing Directory](https://git-scm.com/book/en/v2/Git-Basics-Getting-a-Git-Repository#Initializing-a-Repository-in-an-Existing-Directory)
* [git init docs](https://git-scm.com/docs/git-init)
* [git init Tutorial](https://www.atlassian.com/git/tutorials/setting-up-a-repository)

### Clone An Existing Repo

The git clone command is used to create an identical copy of an existing repository.
```
$ git clone <path-to-repository-to-clone>
```
or if you want to give the directory a name
```
$ git clone <path-to-repository-to-clone> <directory_name>
```

This command:
* takes the path to an existing repository
* by default will create a directory with the same name as the repository that's being cloned
* can be given a second argument that will be used as the name of the directory
* will create the new repository inside of the current working directory

**Helpful Links**:

* [Cloning an Existing Repository](https://git-scm.com/book/en/v2/Git-Basics-Getting-a-Git-Repository#Cloning-an-Existing-Repository)
* [git clone docs](https://git-scm.com/docs/git-clone)
* [git clone Tutorial](https://www.atlassian.com/git/tutorials/setting-up-a-repository)

### Determine A Repo's Status

The git status command will display the current status of the repository.
```
$ git status
```

I can't stress enough how important it is to use this command all the time as you're first learning Git. This command will:
* tell us about new files that have been created in the Working Directory that Git hasn't started tracking, yet
* files that Git _is_ tracking that have been modified

**Helpful Links**:
* [Checking the Status of Your Files](https://git-scm.com/book/en/v2/Git-Basics-Recording-Changes-to-the-Repository#Checking-the-Status-of-Your-Files)
* [git status docs](https://git-scm.com/docs/git-status)
* [git status Tutorial](https://www.atlassian.com/git/tutorials/inspecting-a-repository/git-status)

## Review a Repo's History

### Displaying A Repo's Commits

The `git log` command is used to display all of the commits of a repository.
```
$ git log
```

> By default, `git log` will only show commits for the currently selected branch. It is entirely possible that the commit you're looking for is on another branch. You can view all commits across all branches by executing `git log --branches=*`. The command git branch is used to view and visit other branches. Invoking the command, `git branch -a` will return a list of all known branch names. One of these branch names can then be logged using `git log <branch_name>`.

*By default*, this command displays:
* the SHA
* the author
* the date
* and the message

**Note**: *"By default"* is emphasized, since `git log` command can display a lot more information than just this.

### Navigating The Log

If you're not used to a pager on the command line, navigating in Less can be a bit odd. 

Here are some helpful keys:

* to scroll down, press
  - `j` or `↓` to move down one line at a time
  - `d` to move by half the page screen
  - `f` to move by a whole page screen
* to scroll up, press
  - `k` or `↑` to move _up_ one line at a time
  - `u` to move by half the page screen
  - `b` to move by a whole page screen
* press `q` to quit out of the log (returns to the regular command prompt)

### Changing How Git Log Displays Information

The `--oneline` flag is used to alter how `git log` displays information:
```
$ git log --oneline
```

This command:
* lists one commit per line
* shows the first 7 characters of the commit's SHA
* shows the commit's message

Otherwise you can use the [`git shortlog`](https://git-scm.com/docs/git-shortlog) command. 

This command:
* lists one commit per line
* commits are grouped by author

### Viewing Modified Files

The `--stat` flag ("stat" is short for "statistics") is used to alter how `git log` displays information:
```
$ git log --stat
```

This command:
* displays the file(s) that have been modified
* displays the number of lines that have been added/removed
* displays a summary line with the total number of modified files and lines that have been added/removed

**Note**: The `--stat` flag can be used alongside the `--oneline` flag.

### Viewing File Changes

The `-p` flag (which is the same as the `--patch` flag) is used to alter how `git log` displays information:
```
$ git log -p
```

This command adds the following to the default output:
* displays the files that have been modified
* displays the location of the lines that have been added/removed
* displays the actual changes that have been made

**Annotated git log `-p` Output**:
![Annotated git log `-p` Output](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58a37f65_ud123-l3-git-log-p-lines-removed-annotated/ud123-l3-git-log-p-lines-removed-annotated.png)

Using the image above, let's do a quick recap of the `git log -p` output:

* 🔵 - the file that is being displayed
* 🔶 - the hash of the first version of the file and the hash of the second version of the file
  - not usually important, so it's safe to ignore
* ❤️ - the old version and current version of the file
* 🔍 - the lines where the file is added and how many lines there are
  - `-15,83` indicates that the old version (represented by the `-`) started at line 15 and that the file had 83 lines
  - `+15,85` indicates that the current version (represented by the `+`) starts at line 15 and that there are now 85 lines...these 85 lines are shown in the patch below
* ✏️ - the actual changes made in the commit
  - lines that are red and start with a minus (`-`) were in the original version of the file but have been removed by the commit
  - lines that are green and start with a plus (`+`) are new lines that have been added in the commit
  
**Note**: `git log -p` shows every single file changes, even the lines of code that only have been indented. To ignore whitespace changes use the `-w` flag alongside `-p` flag (so the command will be `git log -p -w`).

### Viewing A Specific Commit

There are two ways to view the changes in a specific commit.

1. The first is by supplying the SHA of the commit you want to see to `git log` as a final argument:
```
$ git log -p fdf5493
```
**Note**: This works with all the `git log` flags we've seen so far.

2. The second way is using a new command `git show`:
```
$ git show fdf5493
```
`git show` can be combined with most of the other flags we've looked at:
* `--stat` - to show the how many files were changed and the number of lines that were added/removed
* `-p` or `--patch` - this the default, but if `--stat` is used, the patch won't display, so pass `-p` to add it again
* `-w` - to ignore changes to whitespace