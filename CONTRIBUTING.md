# Contributing to SiD2ReGenerator

Contributions to SiD2ReGenerator include code, documentation, answering user
questions, running the project's infrastructure, and advocating for all types of
SiD2ReGenerator users.

The SiD2ReGenerator project welcomes all contributions from anyone willing to work in
good faith with other contributors and the community. No contribution is too
small and all contributions are valued.

This guide explains the process for contributing to the SiD2ReGenerator project's core
repository and describes what to expect at each step. Thank you for considering
these point.

Your friendly SiD2ReGenerator guys!

## Jump to sections 
1. [Pull Requests](#pull-requests)
2. [Branches and Commits](#branches-and-commits)
3. [Coding Style](#coding-style)
4. [Examples](#examples)


## Pull Requests

Everybody can propose a pull request (PR). But only the core-maintainers of the
project can merge PR.

The following are the minimal requirements that every PR needs to meet.

- **Pass Continuous Integration (CI)**: Every PR has to pass our CI. This
  includes compilation with a range of compilers and for a range of target
  architectures, passing the unit tests and no detected issues with static code
  analysis tools.

- **Code-Style**: Please consider the [Coding Style](#coding-style) recommendations
  when formatting your code.

- **Separation of Concerns**: Small changes are much easier to review.
  Typically, small PR are merged much faster. For larger contributions, it might
  make sense to break them up into a series of PR. For example, a PR with a new
  feature should not contain other commits with only stylistic improvements to
  another portion of the code.

- **Feature Commits**: The same holds true for the individual PR as well. Every
  commit inside the PR should do one thing only. If many changes have been
  applied at the same time, `git add --patch` can be used to partially stage and
  commit changes that belong together.

- **Commit Messages**: Good commit messages help in understanding changes.
  See the next section.

- **Linear Commit History**: Our goal is to maintain a linear commit history
  where possible. Use the `git rebase` functionality before pushing a PR. Use
  `git rebase --interactive` to squash bugfix commits.

These labels can be used for the PR title to indicate its status.

- `Draft: `: The PR is work in progress and at this point simply informative.
- `[Review]`: The PR is ready from the developers perspective. He requests a
  review from a core-maintainer.
- `[Discussion]`: The PR is a contribution to ongoing technical discussions. The
  PR may be incomplete and is not intended to be merged before the discussion
  has concluded.

The core-maintainers are busy people. If they take especially long to react,
feel free to trigger them by additional comments in the PR thread. Again, small
PR are much faster to review.

It is the job of the developer that posts the PR to rebase the PR on the target
branch when the two diverge.

## Branches and Commits

### Branch names

Name branches with a preceding scope and the change the branch will introduce in the format:

`<scope>/<description>`

Scopes can be:
- feature
- refactor
- patch
- ci
- style
- build

Examples:
```shell

# adds the CEM method as a new feature
git checkout -b feature/CEM-method-for-clustered-drift-detection

# refactors some portion of the code without adding new features or fixing bugs
git checkout -b refactor/CEM-divide-functions-in-smaller-logical-parts

# fixes a specific mistake
git checkout -b patch/CEM-correctly-handle-multidim-concepts-and-data

```

### Commit messages

We have very precise rules over how our git commit messages can be formatted.
This leads to **more readable messages** that are easy to follow when looking
through the **project history**. But also, we use the git commit messages to
**generate the change log**.

This convention is identical to the [Conventional
Commits](https://www.conventionalcommits.org) specification or the one used by
Angular.

Each commit message consists of a **Header**, a **Body** and a **Footer**. The
header has a special format that includes a **Type**, a **Scope** and a
**Subject**:

```text
<type>(<scope>): <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

The **Header** is mandatory and the **Scope** of the header is optional.

Any line of the commit message cannot be longer 100 characters! This allows the
message to be easier to read on GitHub as well as in various git tools.

The footer should contain a 
[closing reference to an issue](https://help.github.com/articles/closing-issues-via-commit-messages/) if any.

Samples: (even more [samples](https://github.com/angular/angular/commits/master))

```text
docs(cem): add documentation for model
```
```text
fix(cem): allow multi dim concepts

Concepts for multidimensional targets can be used now.
```

If the commit reverts a previous commit, it should begin with `revert: `,
followed by the header of the reverted commit. In the body it should say: `This
reverts commit <hash>.`, where the hash is the SHA of the commit being reverted.

The commit **Type** Must be one of the following:

- **build**: Changes that affect the build system or external dependencies
- **ci**: Changes to our CI configuration files and scripts (example scopes:
  travis, appveyor, fuzz)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bug fix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **style**: Changes that do not affect the meaning of the code (white-space,
  formatting, missing semi-colons, etc)
- **test**: Adding missing tests or correcting existing tests

The commit **Scope** is optional, but recommended to be used. It should be the
name of the component which is affected (as perceived by the person reading the
changelog generated from commit messages). The following is the list of
supported scopes:

- **ex**: Example code changes
- **mt**: Changes specifically for multithreading
- **pack**: Packaging setting changes
- **<method-name>**: Changes related to a specific method e.g. **cem**

The **Subject** contains a succinct description of the change:

- Use the imperative, present tense: "change" not "changed" nor "changes"
- Don't capitalize the first letter
- No dot (.) at the end

For the **Body**, Just as in the **Subject**, use the imperative, present tense: "change" not
"changed" nor "changes". The body should include the motivation for the change
and contrast this with previous behavior.

The **Footer** should contain any information about **Breaking Changes** and is also
the place to reference GitHub issues that this commit **Closes**.

## Coding Style

###  General

Some general rules that apply to coding style are highlighted in the following list:

1. Avoid the use of [Magic Numbers](https://en.wikipedia.org/wiki/Magic_number_(programming)#Unnamed_numerical_constants).
2. Avoid using global variables. If you for some reason really need to use
   global variables, prefix them with `g_` so that it is immediately clear in
   the code, that we are dealing with a global variable.
3. Almost all of the code is indented by four (4) spaces and regardless of which
   is better, spaces or tabs, you should indent with four spaces in order to be
   consistent.
4. If you find yourself copy-pasting code, consider refactoring it into a
   function and calling that. Try keeping functions short (about 20-30 lines,
   that way they will in most cases fit onto the screen). In case of simple
   switch statements for example this can be exceeded, but there is not much
   complexity that needs to be understood. Most of the time, if functions are
   longer than 30 lines, they can be refactored into smaller functions that make
   their intent clear.

### Python

Refer to [PEP8](https://peps.python.org/pep-0008) for general python coding style.

Some of the most common rules in the daily coding you would encounter:

- Use `snake_case` for variable, function, method and module/python-file names.
- Use `CamelCase` for class names
- Use `CAPITAL_CASE` for constants that are defined in a module-scope

### Comments and Documentation

#### Comments 

Use comments as follows:

```python
# this is a comment starting on a line, there is exactly one space between the hashtag and the commenting message

def foo():
    print("hello world")  # print hello world, same-line comments are indented 2 spaces behind the commented statement

```

See the following general rules for comments: [\[1\]](https://www.kernel.org/doc/Documentation/CodingStyle)

> Comments are good, but there is also a danger of over-commenting.  NEVER
> try to explain HOW your code works in a comment: it's much better to
> write the code so that the _working_ is obvious, and it's a waste of
> time to explain badly written code.

> Generally, you want your comments to tell WHAT your code does, not HOW.
> Also, try to avoid putting comments inside a function body: if the
> function is so complex that you need to separately comment parts of it,
> you should probably go back to chapter 6 for a while.  You can make
> small comments to note or warn about something particularly clever (or
> ugly), but try to avoid excess.  Instead, put the comments at the head
> of the function, telling people what it does, and possibly WHY it does
> it.

#### Documentation

Use rst format, see [sphinx guide](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html) 
for more examples and details.

```python
"""
[Summary]

:param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
:type [ParamName]: [ParamType](, optional)
...
:raises [ErrorType]: [ErrorDescription]
...
:return: [ReturnDescription]
:rtype: [ReturnType]
"""
```

### Tests

Tests are contained in the `/tests` dir and start with test_ prefix followed by a
module name (corresponding .py file).

   `example: test_cem.py, test_clustering.py`

## Examples

### Dependencies

Add all your dependencies of examples via `poetry add my-dependency --group examples` to the pyproject.toml.

### Notebooks

- Jupyter notebooks should be added as part of the docs. Put them into the folder `docs/source/notebooks`.
- Notebooks file names should be snake_case, e.g. `my_example_notebook.ipynb`.
- Add markdown documentation to your notebooks starting with the title headline in the first cell, e.g.
```md
# My Example Notebook

This is my example notebook
```

### Python file examples

Examples as executable python scripts should be added to `examples/` e.g. `examples/my_example.py` and need to be
executable via `python examples/my_example.py`.

## Still unsure?

If any questions arise concerning code style, feel free to start an issue.
