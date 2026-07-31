# Radar de Viagem Segura

Dashboard desenvolvido em Python e Streamlit para explorar dados históricos de acidentes, comparar localidades e apoiar a escolha de horários de viagem com maior ou menor nível de atenção.

> O sistema é uma ferramenta informativa baseada em dados históricos. Ele não prevê acidentes, não substitui aplicativos de navegação e não dispensa a consulta às condições atuais da via.

## Documentação

[![Baixar documentação em Word](https://img.shields.io/badge/Baixar-Documentação_Word-2563EB?style=for-the-badge&logo=microsoftword&logoColor=white)](./docs/DOCUMENTACAO_COMPLETA_RADAR_VIAGEM_SEGURA.docx?raw=1)

- [Baixar a documentação completa em Word (.docx)](./docs/DOCUMENTACAO_COMPLETA_RADAR_VIAGEM_SEGURA.docx?raw=1)
- [Ler a documentação completa no GitHub](./docs/DOCUMENTACAO_COMPLETA.md)
- [Baixar a versão em Markdown](./docs/DOCUMENTACAO_COMPLETA.md?raw=1)

Para regenerar o arquivo Word após uma atualização da documentação:

```powershell
python -m pip install -r requirements-docs.txt
python scripts/gerar_documentacao_word.py
```

## Funcionalidades

- planejamento de viagem por origem, destino e dia da semana;
- comparação de dois destinos;
- indicação de melhores e piores horários históricos;
- panorama de acidentes, gravidade e distribuição temporal;
- ranking de municípios e rodovias que exigem mais atenção;
- mapa histórico dos registros;
- filtros por fonte, ano e estado;
- consulta e download dos dados filtrados;
- apresentação das métricas do modelo de classificação.

## Execução local

Requisitos: Python 3.11 ou 3.12 e Git.

```powershell
git clone https://github.com/guidsrossi/tcc-vini.git
cd tcc-vini
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Depois, acesse `http://localhost:8501`.

Para recalcular o modelo e suas métricas:

```powershell
python main.py
```

## Estrutura resumida

```text
app.py                 Entrada do dashboard Streamlit
main.py                Pipeline de treinamento e avaliação
dashboard/             Páginas e fluxo do dashboard
src/                    Carga, normalização, risco, modelo e interface
dados/                  Bases CSV utilizadas pelo projeto
modelos/                Modelo treinado
outputs/                Métricas e relatórios gerados
docs/                   Documentação do projeto
```

Consulte a [documentação completa](./docs/DOCUMENTACAO_COMPLETA.md) para instalação detalhada, arquitetura, dados, metodologia, operação e solução de problemas.
