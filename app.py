import streamlit as st
import pandas as pd
from prophet import Prophet  # Importe o Prophet
import json
from prophet.serialize import model_to_json, model_from_json
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Carregue os dados e o modelo
dados = pd.read_csv('https://raw.githubusercontent.com/TatiGandra/tech_challenge_4_Tati/main/df_clean.csv')
dados_total = pd.read_csv('https://raw.githubusercontent.com/TatiGandra/tech_challenge_4_Tati/main/df_total.csv')

# Criar guias
titulos_guias = ['Introdução','Simulador', 'DashBoard', 'Análises Geopolíticas']
guia1, guia2, guia3, guia4 = st.tabs(titulos_guias)

with guia1:
    st.header('Petróleo Brent')
    
    st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)
     
    st.subheader('Introdução')

    st.markdown("""
    
    Bem-vindo a análise dos dados do petróleo brent, que é uma classificação de petróleo cru originário do mar do norte.

    Para este trabalho, importamos a base histórica de preços encontrado na base de dados do Ipea e criamos um simulador de preços deste produto e um dashboard com análises desses mesmos dados e variações geopolíticas que os afetaram.

    Convidamos você a se juntar a nós nesta análise. Acreditamos que você encontrará esta análise informativa e útil. 
    
    Vamos começar!
    """)
    st.subheader('O que é? - Petróleo Brent e WTI')

    st.markdown('''
    Essas duas siglas, que normalmente acompanham a cotação do petróleo, indicam a origem do óleo e o mercado onde ele é negociado. O petróleo Brent foi batizado assim porque era extraído de uma base da Shell chamada Brent. Atualmente, a palavra Brent designa todo o petróleo extraído no Mar do Norte e comercializado na Bolsa de Londres.
    
    A cotação Brent é referência para os mercados europeu e asiático. Já o petróleo WTI tem o nome derivado de West Texas Intermediate. Ele é negociado na Bolsa de Nova York e sua cotação é referência para o mercado norte-americano. 
    
    Nosso trabalho se propõe a analisar apenas o petróleo Brent.
    ''')
    
    st.markdown('<hr style="border-top: 2px solid blue;">', unsafe_allow_html=True)
    
    st.subheader('Pós Tech - FIAP')
    
    st.markdown('Created by:')       
    st.markdown('Leandro Castro - RM 350680')
    st.markdown('Mateus Correa - RM 351094')
    st.markdown('Tatiane Gandra - RM 352177' )
 
    with guia2:
        with open('modelo/prophet_model.json', 'r') as f:
            model_json = json.load(f)
            modelo_carregado = model_from_json(model_json)


        # Permita que o usuário selecione a data desejada
        st.title('Simulador de preço do petróleo brent')
        st.markdown('O simulador do preço do petróleo à data escolhida foi desenvolvido com algoritmo preditivo de machine learning e possui alto grau de confiabilidade dados cenários sem grandes interferências externas.')
        input_data = st.date_input('Selecione uma data', format="YYYY-MM-DD")

        # Crie um DataFrame com a data selecionada pelo usuário
        nova_data = pd.DataFrame({'ds': [input_data]})

        # Faça a previsão para a data selecionada
        previsao = modelo_carregado.predict(nova_data)

        # Exiba o resultado da previsão para o usuário
        if st.button('Enviar'):
            st.write('O preço do petróleo brent previsto para esse dia será de U$', round(previsao['yhat'].iloc[0],2))
            dados['ds'] = pd.to_datetime(dados['ds'])

            # Filtrar os dados para o primeiro trimestre de 2024
            primeiro_trimestre_2024 = dados[(dados['ds'] >= '2024-01-01') & (dados['ds'] <= '2024-03-31')]

            # Calcular a média dos valores do primeiro trimestre de 2024
            media_primeiro_trimestre = primeiro_trimestre_2024['y'].mean()


            # Supondo que 'previsao' é um DataFrame que contém as previsões
            # Supondo que você quer comparar a média do primeiro trimestre com a primeira previsão
            valor_previsto = previsao['yhat'].iloc[0]  # ou o índice que você quiser
            percentual_diferenca = ((valor_previsto - media_primeiro_trimestre) / media_primeiro_trimestre) * 100
            # Comparar a previsão com a média do primeiro trimestre de 2024
            if valor_previsto > media_primeiro_trimestre:
                comparacao = "maior"
            else:
                comparacao = "menor"

            # Formatar a previsão e o percentual com duas casas decimais
            valor_previsto_formatado = round(valor_previsto, 2)
            percentual_diferenca_formatado = round(percentual_diferenca, 2)

            # Exibir a previsão e a comparação
            st.write(f"Ele é {comparacao} do que a média do primeiro trimestre de 2024 ({media_primeiro_trimestre:.2f}). Isto representa uma diferença de {percentual_diferenca_formatado}%.")

        
        with guia3:
            st.header('Time Series - Petróleo')

            # Criar o gráfico de linha
            trace = go.Scatter(x=dados['ds'], y=dados['y'], mode='lines', name='Preço do Petróleo')

            # Criar o layout do gráfico
            layout = go.Layout(title='Time Series - Petróleo', xaxis=dict(title='Data'), yaxis=dict(title='Valor de Fechamento'), hovermode='closest')

            # Criar a figura
            fig = go.Figure(data=[trace], layout=layout)

            # Mostrar o gráfico
            st.plotly_chart(fig)
            st.write('''
                     * Flutuações Significativas: O gráfico mostra uma flutuação significativa nos preços do petróleo, com quedas acentuadas e aumentos notáveis em certos pontos.
                     * Estabilidade e Queda em 2020: Inicialmente, em 2020, o preço estava estável em torno de $60 por barril antes de cair drasticamente.
                     * Volatilidade: Os picos visíveis no gráfico indicam períodos de rápida mudança de preços, refletindo eventos que impactaram o mercado de petróleo.
                     * Mercado Volátil: As flutuações ao longo dos anos mostram um mercado volátil, com variações de preço que podem ser influenciadas por fatores geopolíticos, demanda global e outros fatores econômicos.
                     
                     Este gráfico de linha do tempo é uma ferramenta valiosa para analisar o comportamento do mercado de petróleo ao longo de um período de cinco anos, permitindo identificar tendências e potenciais padrões no movimento dos preços.
                     ''')
            
            st.header('Média Mensal')
            dados['ds'] = pd.to_datetime(dados['ds'])
            dados.set_index('ds', inplace=True)
            # Calcular a média mensal dos preços do petróleo
            monthly_avg = dados.resample('M').mean()

            # Criar o gráfico de barras
            trace = go.Bar(x=monthly_avg.index, y=monthly_avg['y'], marker=dict(color='skyblue'), name='Média Mensal')

            # Criar o layout do gráfico
            layout = go.Layout(title='Média Mensal de Preços do Petróleo', xaxis=dict(title='Data'), yaxis=dict(title='Valor de Fechamento'))

            # Criar a figura
            fig = go.Figure(data=[trace], layout=layout)

            # Mostrar o gráfico
            
            st.plotly_chart(fig)

            st.write('''
                    * Variação Significativa: O gráfico mostra que houve uma variação significativa nos preços ao longo dos cinco anos, com mudanças mensais evidentes.
                    * Picos em 2022: Há picos notáveis em 2022, sugerindo um aumento considerável nos preços médios mensais nesse ano.
                    * Valor da Média Mensal: O eixo Y, rotulado como “Valor da Média Mensal em Reais”, mostra que os preços variaram de 0 a 120 reais.
                    * Análise Temporal: O eixo X, que representa a data de 2020 a 2024, permite visualizar a tendência dos preços ao longo do tempo, destacando meses específicos com preços médios mais altos.

                    Essas informações podem ser cruciais para entender a tendência do mercado de petróleo e para tomar decisões informadas relacionadas a investimentos e estratégias econômicas.
                    '''
            )
            st.header('Histograma')
            # Criar o histograma dos preços do petróleo
            trace = go.Histogram(x=dados['y'], marker=dict(color='skyblue'))

            # Criar o layout do gráfico
            layout = go.Layout(title='Distribuição dos Preços do Petróleo', xaxis=dict(title='Valor de Fechamento'), yaxis=dict(title='Contagem'))

            # Criar a figura
            fig = go.Figure(data=[trace], layout=layout)

            # Mostrar o gráfico
            
            st.plotly_chart(fig)
            st.write('''
                    * Faixa de Preço Mais Comum: A maior frequência de preços está entre 60 a 80 dólares, indicando que esta foi a faixa de preço mais observada.
                    * Variação de Preço: Existem ocorrências significativas na faixa de 80 a 100 dólares, mostrando que os preços também alcançaram valores mais altos com certa regularidade.
                    * Preços Menos Comuns: As faixas de preço abaixo de 20 e acima de 100 dólares têm menos ocorrências, sugerindo que são eventos menos comuns.
                    * Distribuição: O histograma mostra uma distribuição variada dos preços, com a maior parte concentrada na faixa média de preço.
                    
                    O histograma fornece uma visão geral útil da volatilidade dos preços do petróleo e das faixas de preço mais frequentes ao longo dos últimos cinco anos. Isso pode ser útil para análises econômicas e previsões de mercado.
                    '''
            )
            st.header('Boxplot')
            # Criar o boxplot dos preços do petróleo
            trace = go.Box(y=dados['y'], marker=dict(color='skyblue'))

            # Criar o layout do gráfico
            layout = go.Layout(title='Dispersão dos Preços do Petróleo', yaxis=dict(title='Valor de Fechamento'))

            # Criar a figura
            fig = go.Figure(data=[trace], layout=layout)

            # Mostrar o gráfico
            
            st.plotly_chart(fig)
            st.write('''
                    * Mediana: A linha dentro da caixa principal indica a mediana dos preços, que é o valor central da distribuição.
                    * Intervalo Interquartil: A caixa em si representa o intervalo interquartil, mostrando onde a maioria dos dados está concentrada.
                    * Variação dos Preços: Os “bigodes” do boxplot se estendem para mostrar a variação total dos preços dentro de 1,5 vezes o intervalo interquartil.
                    * Uniformidade dos Dados: A ausência de outliers sugere que os preços do petróleo foram relativamente estáveis, sem flutuações extremas.

                    Este boxplot fornece uma visão clara da distribuição dos preços do petróleo, destacando a mediana e a consistência dos preços ao longo do tempo.
            ''')
            st.header('Densidade')
            # Criar o gráfico de densidade dos preços do petróleo
            trace = go.Scatter(x=dados.index, y=dados['y'], mode='lines', fill='tozeroy', fillcolor='skyblue', line=dict(color='blue'))

            # Criar o layout do gráfico
            layout = go.Layout(title='Densidade dos Preços do Petróleo', xaxis=dict(title='Data'), yaxis=dict(title='Valor de Fechamento'))

            # Criar a figura
            fig = go.Figure(data=[trace], layout=layout)

            # Mostrar o gráfico
            
            st.plotly_chart(fig)
            st.write('''
            * Flutuações Visíveis: O gráfico mostra flutuações notáveis nos preços do petróleo, com picos e quedas significativos ao longo do período.
            * Picos Significativos: Entre 2021 e 2022, observam-se picos acentuados, indicando um aumento abrupto nos preços.
            * Variação de Preços: O eixo Y, que representa a ‘Valor da Densidade dos Preços do Petróleo’, varia de 0 a 140, mostrando uma ampla gama de variação nos preços.
            * Análise Temporal: O eixo X cobre de 2020 a 2024, permitindo uma análise da evolução dos preços ao longo do tempo.

            Este gráfico é útil para entender como a densidade dos preços do petróleo mudou e pode ajudar a identificar tendências ou padrões no mercado de petróleo.
            ''')

            with guia4:
                st.header('ANÁLISES GEOPOLÍTICAS E SUA INFLUÊNCIA SOBRE O PREÇO DO PETRÓLEO BRENT')
                st.markdown('Para fazermos as análises geopolíticas, abaixo trazemos um gráfico que nos mostra todas as variações da base histórica.')
                st.subheader('Time Series - Petróleo - Análise desde maio/1987')

                # Criar o gráfico de linha
                trace = go.Scatter(x=dados_total['data_registro'], y=dados_total['preco_venda'], mode='lines', name='Preço do Petróleo')

                # Criar o layout do gráfico
                layout = go.Layout(title='Time Series - Petróleo desde 1987', xaxis=dict(title='Data'), yaxis=dict(title='Valor de Fechamento'), hovermode='closest')

                # Criar a figura
                fig = go.Figure(data=[trace], layout=layout)

                # Mostrar o gráfico
                st.plotly_chart(fig)
                st.subheader('Guerra do Golfo')
                
                st.write(r'''
                Em 1990, o Iraque invade o Kuwait – que participou na Guerra Irã-Iraque. Mais uma vez, uma das mais importantes regiões petrolíferas levanta preocupações no abastecimento do ocidente.
                
                O preço do barril Brent, que no início da Guerra do Golfo, em 2 de agosto de 1990, era cotado a US\$ 22,25, teve um aumento de cerca de 25% ao final do mês.
                
                Já no mês seguinte, apresentava um aumento de 84,27%, chegando a ser cotado a US\$ 41, segundo dados do Energy Information Administration (EIA), divulgados pelo Ipeadata.   
                ''')

                st.subheader('Ataques do 11 de setembro de 2001')
                
                st.write(r'''
                Em 11 setembro de 2001, o mundo assistia aos ataques contra o World Trade Center (WTC), que deixaram cerca de 3 mil mortes.

                A estratégia dos EUA após esse impacto foi de estabelecer uma segurança ao país quanto às exportações de petróleo, sendo a commodity um motim dos conflitos e peça fundamental para as economias.

                Enquanto isso, os preços do petróleo despencaram. No dia dos ataques, o barril Brent era cotado a US\$ 29,12, diminuindo a US\$ 25,57 uma semana depois.

                No final do mês de setembro, a queda foi de quase 25%.
   
                ''')
                st.subheader('Boom das commodities')
                
                st.write(r'''
                O boom das commodities, ou superciclo das commodities, foi um período de forte alta dos preços de grande quantidade de matérias-primas, incluindo alimentos, petróleo, metais e energia. Esse fenômeno ocorreu no início do século XXI, aproximadamente entre 2000 e 2014.

                O preço do barril Brent teve um aumento surpreendente de 560%, com seu valor indo da casa de U\$ 25 a U\$ 140 em 2008.

                Esse aumento nos preços das commodities foi impulsionado principalmente pela crescente demanda das economias emergentes, especialmente da China, bem como por preocupações sobre a disponibilidade desses recursos a longo prazo.

                A única queda expressiva durante este período foi uma consequência da Grande Recessão.
 
                ''')

                st.subheader('Grande Recessão')
                
                st.write(r'''
                A Grande Recessão decorreu do colapso do mercado imobiliário dos Estados Unidos em relação à crise financeira de 2007–2008 e à crise das hipotecas subprime, embora as políticas de outras nações também tenham contribuído. De acordo com o National Bureau of Economic Research, uma organização sem fins lucrativos, a recessão nos Estados Unidos começou em dezembro de 2007 e terminou em junho de 2009, estendendo-se assim por dezenove meses. A Grande Recessão resultou em uma escassez de ativos valiosos na economia de mercado e no colapso do setor financeiro (bancos) na economia mundial.

                A crise financeira e a Grande Recessão induziram uma queda no mercado de petróleo e gás, levando o preço do barril de petróleo brent de quase US\$ 140 para US\$ 45 em apenas seis meses (junho a dezembro de 2008).
   
                ''')

                st.subheader('Pandemia da COVID-19')
                
                st.write(r'''
                Os preços do petróleo caíram drasticamente no início da pandemia de COVID-19, atingindo o seu valor mais baixo desde 2004 que era de U\$ 29. O preço do barril Brent despencou mais de 30%, de US\$ 66 a US\$ 22 de dezembro de a 2019 a março de 2020. Só em outubro de 2021 o valor do barril atingiu novamente a casa dos R\$ 80, valor mais alto desde setembro de 2018.
   
                ''')

                st.subheader('Guerra Rússia-Ucrânia')
                
                st.write(r'''
                Com guerra envolvendo a Rússia foi um pouco diferente, quando a cotação do barril Brent, referência internacional, subiu rapidamente.

                No conflito com a Ucrânia, o crescimento do dia 24 de fevereiro a 3 de março de 2022 foi de 19,21%, com barris cotados em US\$ 118,11.

                Após cerca de 3 meses, os preços caíram. Isso porque, segundo Bassotto, as cadeias produtivas se realocam, e, quem antes comprava da Rússia, migrou para outro mercado ou conseguiu comprar mais barato do país, como Índia e China fizeram.
                ''')