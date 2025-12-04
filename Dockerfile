FROM condaforge/miniforge3:latest

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml && conda clean -afy

ENV PATH="/opt/conda/envs/d100_d300_env/bin:$PATH"

COPY . .

RUN pip install . --no-deps

CMD ["python", "test_setup.py"]
