<h1>Faces em Visão Computacional</h1>

Neste projeto irei abordar 3 aspectos cruciais da tecnologia de reconhecimento facial: a detecção, anonimização e o reconhecimento de faces. 
Estes projetos têm como objetivo garantir a segurança e a privacidade em um mundo cada vez mais digital e interconectado.

<h2>Preparando o ambiente</h2>

~~~
pip install opencv-python
pip install numpy
pip install pickle
pip install face_recognition
pip install os
pip install re
pip install imutils
pip install mediapipe
~~~

Para a execução do mesmo precisamos efetuar a instalação de algumas bibliotecas


Após a instalação destas bibliotecas precisamos criar duas pastas junto ao projeto com os
nomes de "dataset" e "dataset_full". A pasta "dataset" servirá para salvar as fotos tiradas para o reconhecimento facial, já
a pasta "dataset_full" irá atuar com pasta para processamento dos encodings das faces presentes nas fotos.

<h2>Anonimização de Faces</h2>


No projeto Face_Anonymous.py, basta informar qual será a webcam que será utilizada na execução na função "VideoCapture()" atraves de 
uma numeração começando por 0 até a quantidade de cameras que o seu desktop possuir.

Para este projeto tivemos bom resultados, visto que a face presente no teste está bem aparente.

<h2>Detecção de Faces</h2>



Já para a Face_Detection.py, devemos também ajustar o valor da camera a ser utilizada na função "VideoCapture()" e executar o código.

Neste resultado tivemos um ótimo retorno, considerando o alto percentual de confiança da detecção.

<h2>Reconhecimento Facial</h2>

<h3>Captura das Faces</h3>



nesta parte primeiramente precisamos efetuar a captura das faces, ajustando o valor da camera junto ao código como explicado 
anteriormente, iremos executar o mesmo. Durante a sua execução iremos informar o nome da pessoa que será salvo as fotos, depois
podemos pressionar a tecla enter. Na sequencia, o código irá detectar sua face e conforme a tecla "Q" uma foto da face será tirada.
Serão tiradas 20 fotos de cada pessoa antes que o código enecerre sua execução. Este processo pode ser executado várias vezes
para várias pessoas diferentes.

<h3>Treinamento do Detector</h3>

Após as fotos serem tiradas devemos executar o código Encoding_Faces.py, sendo extraido de cada pasta presente no "dataset" seus encodings.

<h3>Reconhecimento Facial</h3>



Já nesta etapa, precisamos informar a camera a ser utilizada e somente exeutar o código Recognition_DeepLearning.py. O baldingbox irá ser
fomado ao redor da face informando o nome da pessoa e sua taxa de confiança na detecção.

Para o resultado final, tivemos também um bom aproveitamento levando em consideração a face aparente e a pouca iluminação durante sua
execução.
