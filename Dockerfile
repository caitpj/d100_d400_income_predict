FROM condaforge/miniforge3@sha256:b176780143fe0b367d71182b06fe19dc67c1faec1ca3f5812f59794d934c8fd1

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

ENV PATH="/opt/conda/envs/d100_d300_env/bin:$PATH"

COPY pyproject.toml .
COPY src/ src/
RUN pip install . --no-deps

COPY . .

CMD ["python", ".test_setup.py"]
