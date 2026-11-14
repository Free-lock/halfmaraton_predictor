````markdown
# 🏃 HalfTime — predyktor czasu półmaratonu

**Od wyniku na 5 km do szacowanego czasu na mecie półmaratonu.**

HalfTime to aplikacja Streamlit wykorzystująca zapisany model
regresyjny PyCaret do przewidywania czasu biegu na dystansie
**21,0975 km**.

Dane można podać ręcznie albo opisać naturalnym językiem.
W drugim przypadku OpenAI pomaga uzupełnić formularz,
a użytkownik sprawdza dane przed wykonaniem predykcji.

> **OpenAI odczytuje dane z opisu. Prognozę czasu oblicza
> osobny, wytrenowany model uczenia maszynowego.**

## ✨ Funkcje

- 📋 Ręczne wprowadzanie danych — bez klucza OpenAI.
- 💬 Odczytywanie danych z opisu w języku polskim.
- ✅ Sprawdzanie i poprawianie danych przed predykcją.
- 🏁 Szacowanie czasu ukończenia półmaratonu.
- ⚡ Wyświetlanie średniego tempa w formacie `M:SS min/km`.
- 📍 Przeliczanie międzyczasów dla biegu w równym tempie.
- 💾 Pobieranie wyniku jako JSON.
- 🔎 Opcjonalne śledzenie analizy AI przez Langfuse.
- ☁️ Możliwość wdrożenia na Streamlit Community Cloud.

## 🧠 Jak to działa?

```text
Opis użytkownika ──→ OpenAI + Instructor ──┐
                                         ↓
Formularz ręczny ─────────────────→ Potwierdzenie danych
                                         ↓
                               Model regresyjny PyCaret
                                         ↓
                            Czas półmaratonu i średnie tempo
```

Model korzysta z trzech cech:

| Cecha | Format |
|---|---|
| Płeć | Kategorie `M` lub `K`, zgodnie ze schematem modelu |
| Wiek | Pełne lata |
| Czas na 5 km | Sekundy |

Interfejs przyjmuje czas jako `MM:SS` lub `HH:MM:SS`
i przelicza go na sekundy.

## 🛠️ Technologie

- Python
- Streamlit
- PyCaret
- Pandas
- OpenAI API
- Pydantic
- Langfuse — opcjonalna obserwowalność

## 📁 Struktura projektu

```text
halftime/
├── app.py
├── halfmarathon_model.pkl
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
└── .streamlit/
    └── config.toml
```

## 📊 Model i wyniki oceny

Aplikacja wykorzystuje model **Gradient Boosting Regressor**
zapisany w pipeline PyCaret.

Model przewiduje czas półmaratonu na podstawie:

- płci zakodowanej jako `M` lub `K`,
- wieku,
- całkowitego czasu biegu na 5 km.


## 🎯 Co przewiduje model?

Model szacuje czas ukończenia półmaratonu na podstawie:

- płci,
- wieku,
- międzyczasu po pierwszych 5 km tego samego biegu.

Zmienną docelową podczas treningu był czas całego półmaratonu.

Aplikacja jest przeznaczona do estymacji wyniku po pokonaniu
pierwszych 5 km. Nie została zwalidowana jako predyktor
przedstartowy wykorzystujący rekord z osobnych zawodów na 5 km.

Raportowane metryki dotyczą zadania opartego na międzyczasach.
Nie należy przenosić ich bez dodatkowej walidacji
na prognozy z niezależnych biegów na 5 km.

### Raportowane metryki

Dla czasu docelowego wyrażonego w sekundach:

| Metryka | Wynik |
|---|---:|
| MAE | 299,32 s — około 4 min 59 s |
| RMSE | 407,11 s — około 6 min 47 s |
| R² | 0,8873 |
| MAPE | 3,92% |
| RMSLE | 0,0515 |
| MSE | 165 735,87 s² |

### Jak interpretować wyniki?

Średni błąd bezwzględny modelu wyniósł około 5 minut
na danych użytych do oceny.

Nie oznacza to, że każda prognoza mieści się w granicach
±5 minut. Błąd dla konkretnego biegacza może być większy.

Wartość R² równa 0,8873 oznacza około 88,73% wyjaśnionej
zmienności na ocenianych danych — nie jest procentem
„poprawnych prognoz”.

MAPE na poziomie 3,92% opisuje średni bezwzględny błąd
procentowy, a nie dokładność klasyfikacji.

Aplikacja nie wyznacza indywidualnego przedziału niepewności.
Precyzyjny zapis wyniku do jednej sekundy nie oznacza
dokładności modelu do jednej sekundy.
## 🚀 Uruchomienie lokalne

### 1. Pobierz projekt

Sklonuj repozytorium lub pobierz jego zawartość,
a następnie przejdź do katalogu projektu.

### 2. Utwórz środowisko wirtualne

Zalecane jest użycie wersji Pythona zgodnej
ze środowiskiem treningowym modelu.

```bash
python -m venv .venv
```

Aktywacja na Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Aktywacja na Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Zainstaluj zależności

```bash
python -m pip install -r requirements.txt
python -m pip check
```

### 4. Dodaj model

Umieść zaufany plik:

```text
halfmarathon_model.pkl
```

w katalogu obok `app.py`.

Model musi przyjmować kolumny:

```text
Płeć
5 km Czas
Wiek
```

Aplikacja zakłada, że wynik modelu jest wyrażony w sekundach.

> Zapisane modele mogą wymagać tych samych wersji bibliotek
> co podczas treningu. Używaj wyłącznie zaufanego pliku modelu.

### 5. Opcjonalnie skonfiguruj API

Skopiuj `.env.example` do `.env` i uzupełnij potrzebne wartości.

### 6. Uruchom aplikację

```bash
streamlit run app.py
```

Domyślny adres lokalny:

```text
http://localhost:8501
```

## 🧭 Korzystanie z aplikacji

### Formularz ręczny

1. Wybierz kategorię płci obsługiwaną przez model.
2. Podaj wiek.
3. Wpisz całkowity czas na 5 km, np. `27:30`.
4. Kliknij **„Oszacuj czas półmaratonu”**.

Ten tryb nie wymaga OpenAI.

### Opis w języku naturalnym

Przykład:

> Mam 43 lata, jestem mężczyzną, a mój czas na 5 km
> wynosi 27 minut i 30 sekund.

1. Dodaj klucz OpenAI, jeśli aplikacja nie ma go skonfigurowanego.
2. Otwórz sekcję uzupełniania formularza z opisu.
3. Wpisz opis i kliknij **„Odczytaj dane”**.
4. Sprawdź oraz popraw odczytane wartości.
5. Zatwierdź formularz, aby obliczyć wynik.

Nie musisz podawać imienia, nazwiska ani informacji medycznych.

## 🔑 Konfiguracja

Przykładowy plik `.env`:

```dotenv
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5-mini

LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

### OpenAI

Klucz jest potrzebny wyłącznie do analizy opisu.
Można podać go w panelu bocznym lub skonfigurować
po stronie aplikacji.

Korzystanie z API jest płatne zgodnie z cennikiem OpenAI.
Abonament ChatGPT nie obejmuje kosztów API.

Dostępność wybranego modelu zależy od konta OpenAI.

### Langfuse

Integracja jest opcjonalna.

Po skonfigurowaniu kluczy użytkownik może wyrazić zgodę
na zapis analizy opisu w projekcie Langfuse administratora.

Śledzenie jest domyślnie wyłączone w interfejsie.
Bez zgody aplikacja używa zwykłego klienta OpenAI.

## 📊 Interpretacja wyniku

Aplikacja prezentuje:

- szacowany czas ukończenia półmaratonu,
- średnie tempo na kilometr,
- międzyczasy przy założeniu stałego tempa.

Międzyczasy są przeliczeniem matematycznym prognozy,
a nie spersonalizowaną rekomendacją taktyki biegu.

Aplikacja nie wyznacza przedziału niepewności.
Precyzja zapisu do jednej sekundy nie oznacza takiej
dokładności modelu.

## ⚠️ Ograniczenia

Prognoza nie jest gwarancją wyniku.

Model korzysta tylko z trzech cech i nie uwzględnia
bezpośrednio m.in.:

- profilu i nawierzchni trasy,
- pogody oraz temperatury,
- przygotowania wytrzymałościowego,
- kilometrażu i historii treningowej,
- odżywiania podczas biegu,
- regeneracji, kontuzji i stanu zdrowia.

Interfejs przyjmuje wiek 13–100 lat.
Jest to ograniczenie aplikacji, a nie potwierdzenie,
że cały ten zakres był reprezentowany w treningu.

Model obsługuje kategorie płci `M/K`, zgodnie
ze swoim schematem wejściowym.

**Wynik nie stanowi porady medycznej ani planu treningowego.**

## 🛡️ Prywatność

- Formularz ręczny nie wymaga wysyłania danych do OpenAI.
- Analiza opisu wysyła jego treść do OpenAI.
- Po włączeniu Langfuse opis i wynik ekstrakcji
  mogą zostać zapisane w systemie obserwowalności.
- Klucz OpenAI nie jest dodawany do eksportowanego wyniku.
- Aplikacja nie zapisuje wyników w swojej bazie danych.
- Wynik jest przechowywany w bieżącej sesji
  i może zostać pobrany jako JSON.

Przy wdrożeniu w chmurze ręczne dane są nadal przetwarzane
na serwerze aplikacji — nie wyłącznie na urządzeniu użytkownika.

Nigdy nie dodawaj `.env` ani `.streamlit/secrets.toml`
do publicznego repozytorium.

## ☁️ Wdrożenie na Streamlit Community Cloud

1. Umieść projekt w repozytorium GitHub.
2. Dodaj przetestowany `requirements.txt`.
3. Zapewnij dostęp do zaufanego pliku modelu.
4. W Streamlit Community Cloud wskaż repozytorium i `app.py`.
5. Wybierz wersję Pythona zgodną z modelem.
6. Opcjonalną konfigurację dodaj w ustawieniach Secrets.

Przykład Secrets:

```toml
OPENAI_MODEL = "gpt-5-mini"

# Opcjonalne:
# OPENAI_API_KEY = "..."

# Opcjonalne:
# LANGFUSE_PUBLIC_KEY = "..."
# LANGFUSE_SECRET_KEY = "..."
# LANGFUSE_BASE_URL = "https://cloud.langfuse.com"
```

> Klucz OpenAI skonfigurowany przez administratora
> umożliwia użytkownikom generowanie kosztów na jego koncie.
> Publiczne wdrożenie wymaga rozważenia logowania,
> limitów użycia i ochrony przed nadużyciami.

## 🧩 Rozwiązywanie problemów

### Nie znaleziono modelu

Sprawdź, czy plik `halfmarathon_model.pkl`
znajduje się obok `app.py`.

### Model nie daje się wczytać

Porównaj wersje PyCaret, scikit-learn, Pandas,
NumPy oraz bibliotek estymatora ze środowiskiem treningowym.

### OpenAI nie odczytuje danych

Sprawdź klucz, dostęp do modelu, limity i środki na koncie.
Możesz też uzupełnić formularz ręcznie.

### Prognoza wygląda nierealistycznie

Sprawdź:

- czy czas na 5 km jest całkowitym czasem, a nie tempem;
- czy model był trenowany na tych samych jednostkach;
- czy zmienna docelowa modelu jest wyrażona w sekundach;
- czy dane użytkownika mieszczą się w zakresie treningowym.

---

🏃 **HalfTime Precictor — prognoza jako wskazówka, nie obietnica wyniku.**
````