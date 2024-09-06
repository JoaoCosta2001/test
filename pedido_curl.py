import os

def fazer_pedido_curl(numero_de_vezes):
    # Comando curl para ser executado
    comando1 = 'curl -X POST http://localhost:8080/predictions/resnet-18 -T gato.jpg'
    comando2 = 'curl -X POST http://localhost:8080/predictions/densenet-121 -T gato.jpg'

    # Executar o comando o número definido de vezes
    for _ in range(numero_de_vezes):
        os.system(comando1)
        os.system(comando2)

# Defina o número de vezes que deseja executar o comando
numero_de_vezes = 5

# Executar o comando
fazer_pedido_curl(numero_de_vezes)
