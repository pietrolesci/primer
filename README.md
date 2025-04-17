## General Intro and Setup

Here we explain how to use the repo.


### Setup

Clone the repo

```bash
git clone https://github.com/codebyzeb/infotokenization.git
```

First, install the [`uv`](https://docs.astral.sh/uv/concepts/projects) environment manager

```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# or simply update it
uv self update
```

Then, install the dependencies. Because of flash attention we do a two step installation process—see [`uv` docs](https://docs.astral.sh/uv/concepts/projects/config/#build-isolation)

```bash
uv sync
uv sync --all-extras
```