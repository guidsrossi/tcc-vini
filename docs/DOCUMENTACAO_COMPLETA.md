# Documentação completa — Radar de Viagem Segura

**Versão:** 1.0  
**Tecnologias principais:** Python, Streamlit, Pandas, Plotly e scikit-learn  
**Repositório:** [github.com/guidsrossi/tcc-vini](https://github.com/guidsrossi/tcc-vini)

## 1. Visão geral

O Radar de Viagem Segura é uma aplicação web para análise exploratória de dados históricos de acidentes de trânsito. O sistema consolida uma base nacional e uma base específica do estado de São Paulo, normaliza seus campos e apresenta indicadores que ajudam o usuário a identificar períodos, municípios e rodovias que historicamente exigiram mais atenção.

A aplicação não calcula rotas em tempo real e não prevê se um acidente acontecerá. Seus resultados devem ser interpretados como referências históricas. Antes de viajar, o usuário deve conferir trânsito, clima, bloqueios, obras e alertas oficiais atualizados.

## 2. Objetivos

### 2.1 Objetivo geral

Facilitar a interpretação de grandes volumes de dados de acidentes por meio de uma interface visual e acessível.

### 2.2 Objetivos específicos

- reunir bases com estruturas diferentes em um schema comum;
- permitir recortes por fonte, ano e unidade federativa;
- apresentar níveis relativos de atenção por horário, município e rodovia;
- apoiar comparações entre destinos;
- disponibilizar tabelas, gráficos, mapas e exportação de recortes;
- treinar e avaliar um modelo de classificação de acidentes graves;
- comunicar limitações e nível de suporte dos dados apresentados.

## 3. Público-alvo

- motoristas interessados em consultar padrões históricos;
- estudantes e pesquisadores de mobilidade e segurança viária;
- professores e avaliadores do trabalho acadêmico;
- analistas que desejem explorar as bases consolidadas.

## 4. Funcionalidades da aplicação

### 4.1 Planejar viagem

Permite informar origem, destino e dia da semana. O destino é procurado nos municípios presentes na base selecionada. A tela apresenta uma nota de atenção, faixa de risco relativa, horários de referência, justificativas baseadas nos dados e um link para abrir a rota no Google Maps.

Também é possível comparar dois destinos lado a lado. O histórico recente da sessão facilita repetir consultas já realizadas.

### 4.2 Panorama rápido

Apresenta indicadores agregados, distribuição por horário e evolução anual. Serve como visão inicial do recorte definido nos filtros globais.

### 4.3 Lugares com mais atenção

Exibe rankings históricos de municípios e rodovias. Os resultados consideram volume, gravidade e suporte amostral, evitando interpretar apenas a contagem bruta como risco absoluto.

### 4.4 Melhores horários

Compara as 24 horas do dia para o dia da semana escolhido. A aplicação destaca janelas historicamente mais tranquilas e períodos que exigem mais atenção.

### 4.5 Mapa histórico

Representa geograficamente registros que possuem latitude e longitude válidas. O mapa descreve a distribuição dos dados disponíveis, e não a situação atual das vias.

### 4.6 Tabela

Mostra os registros consolidados após a aplicação dos filtros. O usuário pode baixar o recorte em CSV para análises externas.

### 4.7 Sobre

Resume a metodologia, a composição das bases e as métricas técnicas armazenadas pelo pipeline de modelagem.

## 5. Filtros globais

- **Fonte:** Todas, Base nacional ou Base SP;
- **Ano:** anos identificados a partir da data do acidente ou do nome do arquivo;
- **Estado:** unidades federativas disponíveis para a fonte selecionada.

Os filtros são aplicados antes da geração dos indicadores de cada página.

## 6. Arquitetura

```text
Navegador
   |
   v
Streamlit (app.py)
   |
   +--> dashboard/dashboard.py  -> páginas, filtros e navegação
   +--> src/ui.py               -> componentes visuais e gráficos
   +--> src/risk_engine.py      -> indicadores e referências históricas
   +--> src/data_loader.py      -> descoberta, leitura e normalização
   +--> src/modeling.py         -> treinamento, avaliação e artefatos
   |
   +--> dados/*.csv
   +--> modelos/modelo_risco.pkl
   +--> outputs/model_metrics.json
```

### 6.1 Arquivos principais

| Caminho | Responsabilidade |
|---|---|
| `app.py` | Ponto de entrada do Streamlit. |
| `main.py` | Executa carga, validação, engenharia de atributos, treinamento e persistência. |
| `dashboard/dashboard.py` | Organiza páginas, cache, filtros e interação do dashboard. |
| `src/config.py` | Centraliza caminhos, textos, páginas, fontes e cores. |
| `src/data_loader.py` | Lê CSVs, detecta separador/codificação, normaliza e valida as bases. |
| `src/risk_engine.py` | Calcula componentes de atenção e agregações por tempo e local. |
| `src/modeling.py` | Prepara dados, cria pipeline, treina, avalia e salva o modelo. |
| `src/ui.py` | Define CSS, cartões, gráficos, navegação e componentes reutilizáveis. |
| `src/utils.py` | Contém normalização textual, formatação e conversões numéricas. |

## 7. Estrutura do repositório

```text
.
├── .streamlit/
│   └── config.toml
├── dados/
│   ├── acidentes2025_sp_der_todas_causas_tipos.csv
│   ├── acidentes2025_todas_causas_tipos_parte_001.csv
│   ├── ...
│   └── acidentes2026_todas_causas_tipos.csv
├── dashboard/
│   └── dashboard.py
├── docs/
│   └── DOCUMENTACAO_COMPLETA.md
├── modelos/
│   └── modelo_risco.pkl
├── outputs/
│   ├── model_metrics.json
│   └── relatorio_metricas.txt
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── modeling.py
│   ├── risk_engine.py
│   ├── ui.py
│   └── utils.py
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

O diretório `.venv` é local e não faz parte do repositório. Ele é recriado a partir do `requirements.txt`.

## 8. Dados

### 8.1 Descoberta automática

O carregador procura arquivos CSV diretamente em `dados/`. Arquivos cujo nome contém referências a SP e DER são classificados como **Base SP**. Os demais arquivos de acidentes com ano no nome são classificados como **Base nacional**.

### 8.2 Arquivo nacional dividido

O arquivo nacional de 2025 ultrapassava o limite de 100 MB por arquivo do GitHub. Ele foi dividido em partes de aproximadamente 45 MB, nomeadas com o sufixo `_parte_NNN`.

O carregador remove esse sufixo ao formar `arquivo_origem` e `uid_acidente`. Dessa forma, as partes continuam sendo tratadas como uma única fonte lógica e a divisão física não altera os identificadores apresentados pelo sistema.

### 8.3 Leitura

Para cada CSV, o sistema:

1. tenta as codificações UTF-8 e Latin-1;
2. detecta o separador entre ponto e vírgula, vírgula, tabulação e barra vertical;
3. normaliza nomes de colunas;
4. converte números, datas e horários;
5. agrega linhas da base nacional pelo identificador do acidente;
6. converte a base DER-SP para o schema canônico;
7. concatena as fontes;
8. valida as colunas obrigatórias.

### 8.4 Schema canônico mínimo

| Campo | Descrição |
|---|---|
| `uid_acidente` | Identificador estável formado pela fonte lógica e pelo ID original. |
| `data_inversa` | Data do acidente. |
| `uf` | Unidade federativa. |
| `municipio` | Município normalizado em letras maiúsculas. |
| `hora` | Hora entre 0 e 23. |
| `fonte_base` | Base nacional ou Base SP. |
| `feridos_leves` | Quantidade de feridos leves. |
| `feridos_graves` | Quantidade de feridos graves. |
| `mortos` | Quantidade de mortos. |
| `acidente_grave` | Indicador derivado de morte ou ferimento grave. |
| `rodovia` | Identificação da rodovia quando disponível. |
| `ano_referencia` | Ano obtido da data ou do nome do arquivo. |

Outros campos, como clima, tipo de pista, traçado, causa, latitude e longitude, são preservados quando disponíveis.

## 9. Metodologia dos indicadores históricos

O módulo `risk_engine.py` acrescenta atributos derivados e produz referências agregadas. Entre os sinais utilizados estão:

- quantidade de registros;
- mortos e feridos graves;
- proporção de acidentes graves;
- condições meteorológicas adversas;
- causas comportamentais;
- características de rodovia;
- período do dia;
- volume e cobertura amostral.

Os valores são normalizados para permitir comparação dentro do recorte consultado. O suporte amostral influencia a apresentação e a confiança: resultados com poucos registros devem ser interpretados com maior cautela.

### 9.1 Interpretação da nota

A nota de 0 a 100 é uma medida relativa de atenção dentro dos dados históricos selecionados. Quanto maior a nota, mais sinais históricos desfavoráveis foram encontrados. Ela não representa probabilidade matemática de acidente e não pode ser comparada diretamente a estatísticas oficiais de exposição por quilômetro percorrido.

### 9.2 Limitações metodológicas

- as bases podem possuir subnotificação, campos ausentes ou critérios distintos;
- mais acidentes em um local podem refletir maior fluxo de veículos;
- não há denominador de exposição, como veículos/dia ou quilômetros percorridos;
- o histórico não incorpora automaticamente clima, obras e bloqueios em tempo real;
- correlação histórica não demonstra causalidade;
- rankings dependem dos filtros e da cobertura da base escolhida.

## 10. Modelo de classificação

O pipeline em `src/modeling.py` prepara até 100 mil registros, cria atributos temporais e treina um classificador para o alvo `acidente_grave`.

### 10.1 Atributos utilizados

- UF, município, rodovia e fonte;
- hora, mês, dia da semana, fim de semana e período do dia;
- causa e tipo do acidente;
- condição meteorológica, tipo de pista e traçado da via.

O pipeline aplica transformações adequadas a campos categóricos, numéricos e cíclicos. Quando existem anos suficientes, a avaliação usa separação temporal: anos anteriores no treino e o ano mais recente no teste.

### 10.2 Métricas atualmente armazenadas

| Métrica | Valor aproximado |
|---|---:|
| Accuracy | 0,583 |
| Precision | 0,356 |
| Recall | 0,709 |
| F1-score | 0,474 |
| Limiar | 0,580 |
| Amostras de treino/avaliação registradas | 49.999 |

Na avaliação temporal armazenada, 2025 foi utilizado no treino e 2026 no teste. As métricas podem mudar quando o comando `python main.py` for executado novamente com dados diferentes.

O recall foi priorizado em relação à precision para capturar uma parcela maior dos acidentes graves, ao custo de mais falsos positivos. O modelo é experimental e não deve ser utilizado como único fundamento para decisões de segurança.

## 11. Requisitos

- Windows, Linux ou macOS;
- Python 3.11 ou 3.12 recomendado;
- Git;
- memória disponível compatível com a leitura das bases; recomenda-se pelo menos 4 GB de RAM livre;
- navegador moderno.

Dependências Python declaradas:

- pandas;
- numpy;
- scikit-learn;
- streamlit;
- joblib;
- openpyxl;
- plotly.

## 12. Instalação no Windows

```powershell
git clone https://github.com/guidsrossi/tcc-vini.git
cd tcc-vini
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Se o PowerShell bloquear a ativação, é possível executar o Python do ambiente diretamente:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## 13. Instalação no Linux ou macOS

```bash
git clone https://github.com/guidsrossi/tcc-vini.git
cd tcc-vini
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 14. Execução

Com o ambiente ativado:

```powershell
streamlit run app.py
```

Ou sem ativá-lo, no Windows:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Acesse `http://localhost:8501`. A primeira carga pode demorar alguns segundos porque centenas de milhares de linhas dos arquivos de origem precisam ser lidas e normalizadas.

Para limitar o servidor ao próprio computador:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address=127.0.0.1
```

## 15. Treinamento e atualização dos artefatos

```powershell
.\.venv\Scripts\python.exe main.py
```

O comando:

1. carrega e valida todas as bases;
2. adiciona os atributos de risco;
3. prepara os dados de treinamento;
4. treina e avalia o pipeline;
5. salva `modelos/modelo_risco.pkl`;
6. atualiza `outputs/model_metrics.json`;
7. atualiza `outputs/relatorio_metricas.txt`.

Antes de publicar novos artefatos, revise as métricas e confirme que a mudança de dados não introduziu regressões.

## 16. Testes e verificações

Verificação rápida existente:

```powershell
.\.venv\Scripts\python.exe smoke_test.py
```

Validação direta da carga:

```powershell
.\.venv\Scripts\python.exe -c "from src.data_loader import load_accident_data, validate_loaded_data; print(validate_loaded_data(load_accident_data()))"
```

O resultado deve indicar `ok: True` e nenhuma coluna obrigatória ausente.

## 17. Uso do dashboard

1. abra a aplicação;
2. escolha a página na navegação superior;
3. selecione fonte, anos e estados;
4. aguarde a atualização do recorte;
5. consulte gráficos e indicadores;
6. na página de planejamento, informe o município sem necessidade de acentuação exata;
7. na página de tabela, baixe o recorte em CSV quando necessário.

## 18. Cache e desempenho

O Streamlit mantém em memória os resultados das funções decoradas com `st.cache_data`. O dashboard também tenta usar `outputs/base_cache.parquet` quando o suporte a Parquet está disponível. Esse arquivo é gerado localmente e ignorado pelo Git.

Boas práticas:

- aguardar a primeira carga antes de atualizar a página;
- evitar iniciar duas instâncias na mesma porta;
- fechar servidores antigos antes de reiniciar;
- manter somente uma cópia de cada parte da base;
- não versionar `.venv`, caches Python ou logs.

## 19. Solução de problemas

### Página abre, mas permanece vazia

- aguarde a primeira carga;
- pressione `Ctrl + F5`;
- verifique se há mais de um servidor na porta 8501;
- encerre as instâncias antigas e inicie novamente;
- execute o `smoke_test.py` para verificar a carga.

### `python` não é reconhecido

Instale Python 3.12 e marque a opção de adicioná-lo ao PATH. No Windows, tente também o caminho completo do executável dentro de `.venv`.

### Dependência ausente

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Erro de porta ocupada

Use outra porta:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port=8502
```

### GitHub rejeita arquivo grande

Nenhum arquivo enviado pode exceder 100 MB. A base nacional de 2025 já está dividida em partes menores. Não recombine essas partes dentro do repositório.

### Git mostra milhares de arquivos

Confirme que `.venv/`, `__pycache__/`, `*.pyc` e `*.log` permanecem no `.gitignore`. Instalações locais nunca devem ser commitadas.

## 20. Publicação no GitHub

```powershell
git status
git add .
git commit -m "Atualiza projeto e documentação"
git push -u origin main
```

Sempre revise `git status` antes do commit. As bases divididas, o código, o modelo e os relatórios necessários podem ser versionados; ambiente virtual, caches, logs e segredos não podem.

## 21. Segurança e privacidade

- não adicione senhas, tokens ou chaves ao repositório;
- use `.streamlit/secrets.toml` somente localmente; ele está ignorado;
- ao executar localmente, prefira `--server.address=127.0.0.1`;
- revise os CSVs antes de incluir novas fontes;
- não exponha o dashboard à internet sem autenticação e avaliação de segurança.

## 22. Manutenção e inclusão de novas bases

Para incluir um novo CSV:

1. coloque o arquivo em `dados/`;
2. use um nome que contenha `acidentes` e o ano;
3. para uma base DER-SP, inclua `sp` e `der` no nome;
4. confirme separador, codificação e cabeçalhos;
5. execute a validação da carga;
6. revise quantidade de linhas, anos, fontes e UFs;
7. execute os testes;
8. retreine o modelo, se apropriado;
9. confirme que nenhum arquivo excede 100 MB.

Alterações de schema devem ser implementadas primeiro em `src/data_loader.py`, preservando as colunas canônicas utilizadas pelo restante do sistema.

## 23. Limitações e uso responsável

Esta aplicação é um protótipo acadêmico. Seus resultados não constituem recomendação oficial, garantia de segurança, previsão de acidente ou orientação de emergência. O usuário continua responsável por respeitar a legislação, avaliar suas condições de direção e consultar fontes oficiais.

Em situação de emergência, procure os serviços públicos competentes. No Brasil, os principais números são 190 (Polícia Militar), 191 (Polícia Rodoviária Federal), 192 (SAMU) e 193 (Corpo de Bombeiros).

## 24. Glossário

| Termo | Significado no projeto |
|---|---|
| Acidente grave | Registro com pelo menos um morto ou ferido grave. |
| Base nacional | Conjunto nacional normalizado pelo carregador. |
| Base SP | Conjunto específico do DER de São Paulo. |
| Nota de atenção | Indicador relativo de 0 a 100 calculado com sinais históricos. |
| Suporte | Quantidade e cobertura de dados por trás de um resultado. |
| Schema canônico | Estrutura comum usada depois da normalização das fontes. |
| Holdout temporal | Avaliação em que o período mais recente é separado para teste. |

## 25. Licença e autoria

Antes de redistribuir ou reutilizar o projeto e suas bases, adicione ao repositório uma licença explícita e confirme os termos de uso das fontes de dados. Na ausência de um arquivo de licença, os direitos de reutilização não são concedidos automaticamente.

---

Documento mantido junto ao código. Ao alterar dados, metodologia, dependências, telas ou procedimentos de execução, atualize esta documentação no mesmo commit.
