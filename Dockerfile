FROM condaforge/miniforge3:latest

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

ENV PATH="/opt/conda/envs/d100_d400_env/bin:$PATH"

COPY pyproject.toml .
COPY src/ src/
RUN pip install -e . --no-deps

COPY . .

CMD ["python", ".test_setup.py"]
