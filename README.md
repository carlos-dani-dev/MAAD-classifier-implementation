# MAAD-classifier-implementation
<p>ENG: MAAD Classifier, the MAC (Massive Attribute Classifier), implementation</p>
<p>POR-BR: Implementação do Classificador do MAAD, o MAC (Massive Attribute Classifier)</p>

### Paper relacionado
<p>O paper que idealizou esse classificador pode ser encontrado em <a href='https://arxiv.org/abs/2012.01030'>MAAD-Face: A Massively Annotated Attribute Dataset for Face Images</a>.</p>
<p>Os autores utilizaram este classificador para TRANSFERIR ANOTAÇÕES DE UM OU MAIS DATASETs para um dataset alvo e, a partir isso, montaram o MAAD descrito no paper acima.</p>
<p>O classificador, no entanto, é  apenas 1 parte do PIPELINE INOVADOR proposto no paper. Outros passos como cálculo de confiança preditiva e agregação de anotações também são performados a partir dos resultados de predição do MAC.</p>
<p>Além disso, analisou-se o impacto da presença de determinados rótulos em diferentes subsets de treinamento e como um modelo de reconhecimento pode ser influenciado por isso.</p>

### Funcionamento do MAC
<p>O funcionamento completo do pipeline e do classificador do MAC pode ser visualizado no slide <a href='https://www.canva.com/design/DAGHGZHBG2M/RUtsHfD4URyNCa3Z_z4rjg/edit?utm_content=DAGHGZHBG2M&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton'>Classificador MAC</a></p>

### Checklist da produção
<ul>
  <li><s>Implementação do pipeline de obtenção de FaceNet Embeddings</s></li>
  <li><s>Implementação do algoritmo de Machine Learning do MAC</s></li>
  <li>Implementação do pipeline de cálculo de confiabilidade das previsões do MAC</li>
  <li>Determinação do threshold de confiabilidade para cada atributo</li>
  <li>Descarte de predições a partir do threshold de confiabilidade</li>
  <li>Agregação dos atributos restantes no dataset alvo</li>
</ul>
