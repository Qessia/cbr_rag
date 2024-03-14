# CBR RAG
## Контекст
It Purple Hack хакатон. Q&A; Кейс от Центробанка

## Описание
Проект позволяет использовать RAG LLM систему для ответа на вопросы по документам сайта Центробанка (`cbr.ru`) с помощью удобного интерфейса Telegram-бота.

## Стек технологий
При разработке были использованы:
- `Clickhouse` в качестве базы данных для хранения embedding'ов
- Модель на базе `BERT` для embedding'ов (`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`)
- Пайплайн `PCA` для понижения размерности embedding'ов
- `Mistral7B-Instruct` в качестве LLM

## Использование
Начать общение с Telegram ботом: `@cbr_purple_bot`
