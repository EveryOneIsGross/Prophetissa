# Prophetissa
RAG dataset generator using ollama and emo vector search.
uses fine tuning prompt template from : [mistral ft guide](https://github.com/mistralai/mistral-finetune)


```mermaid

graph TD
A((A)) --> B((B))
B((B)) --> C((C))
B((B)) --> D((D))
B((B)) --> E((E))
C((C)) --> F((F))
D((D)) --> G((G))
E((E)) --> H((H))
F((F)) --> I((I))
G((G)) --> I((I))
H((H)) --> I((I))
I((I)) --> J((J))
I((I)) --> K((K))
I((I)) --> L((L))
J((J)) --> M((M))
K((K)) --> M((M))
L((L)) --> M((M))
M((M)) --> N((N))
N((N)) --> O((O))
O((O)) --> P((P))
P((P)) --> Q((Q))
Q((Q)) --> A((A))

R((R)) --> C((C))
S((S)) --> D((D))
T((T)) --> E((E))
U((U)) --> F((F))
V((V)) --> G((G))
W((W)) --> H((H))
X((X)) --> I((I))

subgraph Input
A((A))
end

subgraph Preprocess
B((B))
C((C))
D((D))
E((E))
end

subgraph TrainWord2Vec
F((F))
G((G))
H((H))
end

subgraph AnalyzeSentiment
I((I))
end

subgraph SmoothVectors
J((J))
K((K))
L((L))
end

subgraph SemanticSearch
M((M))
N((N))
end

subgraph DensityMapping
O((O))
P((P))
end

subgraph AdaptiveChunking
Q((Q))
end

subgraph Output
R((R))
S((S))
T((T))
U((U))
V((V))
W((W))
X((X))
end

subgraph "Parameter Space"
R((R))
S((S))
T((T))
U((U))
V((V))
W((W))
X((X))
end
```
