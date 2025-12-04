FROM condaforge/miniforge3:latest

WORKDIR /app

COPY environment.yml .

# Create the environment
RUN conda env create -f environment.yml && conda clean -afy

# Add the new environment to the PATH
ENV PATH /opt/conda/envs/d100_d300_env/bin:$PATH

# Copy the rest of the code (scripts, etc.)
COPY . .

# Test the installation
CMD ["python", "test_setup.py"]