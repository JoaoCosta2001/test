# Use a imagem oficial do PyTorch como base
FROM pytorch/torchserve:0.6.0-cpu

# Switch to root user to install packages
USER root

# Instala dependências adicionais (se necessário)
RUN apt-get update && apt-get install -y \
    git \
    && apt-get clean

# Cria um diretório para o modelo
RUN mkdir -p /home/model-server/

# Define o diretório de trabalho
WORKDIR /home/model-server/

# Copia o arquivo de configuração e o script de custom service handler
COPY config.properties .
COPY model-store/resnet-18.mar model-store/
COPY model-store/densenet-121.mar model-store/
COPY handler.py .

# Exponha a porta 8080 para o TorchServe
EXPOSE 8080

# Comando para iniciar o TorchServe com o modelo
CMD ["torchserve", "--start", "--ncs", "--ts-config", "config.properties"]