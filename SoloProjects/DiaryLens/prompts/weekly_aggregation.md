# Weekly aggregation prompt v3.3

Ты агрегируешь несколько **daily JSON** в один **weekly JSON** для проекта DiaryLens.

Это техническая задача структурирования недели.

Не пиши финальный markdown-отчёт.  
Не давай советы в свободном тексте.  
Не делай психологический разбор.  
Не добавляй факты, которых нет в daily JSON.

Твоя задача — собрать из daily JSON доказательную недельную структуру:

- главные фактические блоки недели;
- wins;
- tensions;
- emotional background;
- body / energy signals;
- study / project signals;
- social context;
- actual focus;
- repeated topics;
- contradictions;
- open loops;
- risks;
- next week focus candidates;
- what not to do.

Главный принцип:

```text
daily JSONs → day-by-day map → weekly blocks → evidence-backed claims → weekly JSON
```

Не начинай с красивого обобщения недели.  
Сначала восстанови карту недели по дням.

---

## Входные данные

- week_id: {week_id}
- start_date: {start_date}
- end_date: {end_date}
- days_included: {days_included}
- missing_days: {missing_days}

Daily JSONs:

{daily_jsons}

---

# 1. Источник истины

Источник истины — только переданные daily JSONs.

Не используй исходные markdown-файлы дневника, если они не переданы во входе.  
Не добавляй факты, причины, эмоции, диагнозы или выводы, которых нет в daily JSON.

Daily JSON может быть неидеальным, но weekly aggregation не должна усиливать его ошибки.

Если daily JSON содержит очевидно сомнительную классификацию, например игра попала в `wins`, но рядом есть tension “от скуки” или “не самое лучшее решение”, не используй это как weekly win.

---

# 2. Жёсткий контракт Pydantic

Ответ будет валидироваться через Pydantic.

Поэтому нельзя пропускать обязательные поля.

## 2.1. Top-level JSON

`days_included` и `missing_days` обязательны и должны быть скопированы из входных данных без изменения.

Запрещено:
- сокращать `days_included` до одного дня;
- удалять дни из `days_included`, если они есть во входных данных;
- оставлять `missing_days: []`, если ты сам убрал дни из `days_included`;
- делать weekly JSON по одному дню, если во входе переданы несколько daily JSON.

Если во входе `days_included` содержит 7 дат, в ответе `days_included` тоже должен содержать эти 7 дат.

Top-level поле `short_summary` обязательно.

Всегда добавляй его последним полем JSON.

Если не можешь сформулировать хороший summary, всё равно верни:

```json
"short_summary": ""
```

## 2.2. Evidence object

Каждый evidence object ОБЯЗАН содержать ровно эти поля:

```json
{
  "date": "YYYY-MM-DD",
  "source_day_md": "data/processed/days_md/YYYY-MM-DD.md",
  "source_daily_json": "data/processed/days_json/YYYY-MM-DD.json",
  "source_field": "important_moments",
  "quote": "...",
  "note": "..."
}
```

Правила:

- `date` обязателен.
- `source_day_md` обязателен.
- `source_daily_json` обязателен.
- `source_field` обязателен.
- `quote` обязателен.
- `note` обязателен. Если пояснение не нужно, используй `null`.
- Не сокращай evidence object.
- Не делай evidence object только из `date`, `quote`, `note`.
- Не пропускай `source_day_md` ради краткости.
- Не пропускай `source_daily_json` ради краткости.

Если `source_day_md` отсутствует во входном daily JSON, восстанови его по шаблону:

```text
data/processed/days_md/YYYY-MM-DD.md
```

Если `source_daily_json` отсутствует во входном daily JSON, восстанови его по шаблону:

```text
data/processed/days_json/YYYY-MM-DD.json
```

Например:

```json
{
  "date": "2026-05-26",
  "source_day_md": "data/processed/days_md/2026-05-26.md",
  "source_daily_json": "data/processed/days_json/2026-05-26.json",
  "source_field": "tensions",
  "quote": "не самое лучшее решение",
  "note": "Автор сам оценивает игру как сомнительное решение."
}
```

## 2.3. Разрешённые `source_field`

`source_field` может быть только одним из:

- `important_moments`
- `wins`
- `tensions`
- `emotional_signals`
- `body_energy_signals`
- `study_signals`
- `ml_ds_signals`
- `social_signals`
- `decisions`
- `open_loops`
- `key_quotes`
- `short_summary`

Запрещено использовать любые другие значения.

Особенно запрещено:

- `emotions`
- `problems`
- `energy_signals`
- `health_signals`
- `open_questions`
- `body`
- `energy`
- `facts`
- `quotes_or_evidence`
- `professional/project_discussion`
- `project_discussion`
- `professional_discussion`
- `project`
- `work`
- `work_project`
- `professional`

Никогда не придумывай новое значение `source_field` по смыслу пункта.

`source_field` — это НЕ тип события и НЕ тема evidence.  
`source_field` — это только одно из 12 canonical значений Pydantic enum:

```text
important_moments
wins
tensions
emotional_signals
body_energy_signals
study_signals
ml_ds_signals
social_signals
decisions
open_loops
key_quotes
short_summary
```

Если evidence взята из project-related item внутри `important_moments`, используй:

```json
"source_field": "important_moments"
```

Если evidence взята из `study_signals`, используй:

```json
"source_field": "study_signals"
```

Если evidence взята из `ml_ds_signals`, используй:

```json
"source_field": "ml_ds_signals"
```

## 2.4. Нормализация daily fields в canonical source_field

Daily JSON может использовать старые или промежуточные названия полей.

При переносе в weekly evidence используй canonical `source_field` из Pydantic-схемы:

| Поле в daily JSON | `source_field` в weekly evidence |
|---|---|
| `important_moments` | `important_moments` |
| `wins` | `wins` |
| `problems` | `tensions` |
| `tensions` | `tensions` |
| `emotions` | `emotional_signals` |
| `emotional_signals` | `emotional_signals` |
| `energy_signals` | `body_energy_signals` |
| `health_signals` | `body_energy_signals` |
| `body_energy_signals` | `body_energy_signals` |
| `study_signals` | `study_signals` |
| `ml_ds_signals` | `ml_ds_signals` |
| `social_signals` | `social_signals` |
| `decisions` | `decisions` |
| `open_questions` | `open_loops` |
| `open_loops` | `open_loops` |
| `key_quotes` | `key_quotes` |
| `short_summary` | `short_summary` |

Пример:

Если daily JSON содержит:

```json
"emotions": [
  {
    "quote": "мне было капец как скучно",
    "note": "Чувство скуки."
  }
]
```

то в weekly evidence нужно писать:

```json
"source_field": "emotional_signals"
```

а не:

```json
"source_field": "emotions"
```

---

# 3. Обязательный порядок работы

Перед генерацией final weekly JSON мысленно сделай 6 проходов.

Не выводи эти проходы.  
Верни только final JSON.

---

## Проход 1. Day-by-day map

Сначала построи внутреннюю карту недели по каждому дню из `days_included`.

Для каждого дня определи:

1. главные фактические блоки дня;
2. wins;
3. tensions / problems;
4. emotions;
5. body / energy / health signals;
6. study / project signals;
7. social signals;
8. decisions;
9. open questions;
10. key quotes;
11. short summary.

Эту карту не выводи.

Но каждый included day должен быть рассмотрен.

День не должен исчезнуть из weekly JSON, если в нём есть сильный сигнал.

Сильный сигнал — это:

- win;
- tension / problem;
- emotion;
- open question;
- decision;
- key quote;
- major event;
- study / project signal;
- social event;
- body / energy / health signal;
- заметный фактический блок дня.

Если день не представлен ни в одном weekly-поле, это допустимо только если он действительно не содержит значимых weekly signals.

---

## Проход 2. Weekly block inventory

После day-by-day map сгруппируй неделю в фактические блоки.

Проверь минимум такие группы:

- work / commute / money;
- study / projects / ML / DS;
- sport / body / health / energy;
- games / content / entertainment;
- social activity;
- household / money / repairs;
- boredom / emptiness / sluggishness;
- goals / planning / life direction;
- recovery / rest;
- decisions / open loops.

Не все группы обязаны быть заполнены.

Но если группа встречается в 2+ днях или имеет сильный единичный сигнал, она должна быть рассмотрена в weekly JSON.

---

## Проход 3. Field-by-field aggregation

Заполни weekly поля из day-by-day map и weekly block inventory.

---

### `week_essence`

Это осторожная суть недели.

`week_essence` должна опираться минимум на 4 разных дня из 7, если все 7 дней включены.

Если evidence меньше 4 дней, запрещено писать:

- “неделя прошла преимущественно…”;
- “главный фокус недели…”;
- “вся неделя была…”;
- “основная тема недели…”;
- “неделя была в основном…”.

В таком случае формулируй осторожно:

- “одной из заметных тем было…”;
- “в части недели проявлялось…”;
- “несколько дней были связаны с…”.

Для полной недели `week_essence` обычно должна отражать 3–5 фактических блоков:

- что реально происходило;
- что повторялось;
- какие напряжения были заметны;
- какой был общий характер недели.

Не делай `week_essence` красивым философским выводом.

---

### `main_events`

Главные фактические события и блоки недели.

Сюда должны попадать:

- работа / дорога, если занимали заметное место;
- учёба / лабы / проекты, если были;
- спорт / тело / здоровье, если были;
- бытовые и финансовые события;
- игры / контент / отдых, если занимали заметное место;
- социальные события;
- дни с сильным внутренним напряжением, если это важно для недели.

Если есть 2+ рабочих дня, рабочий блок должен быть отражён.

Если есть 2+ социальных дня, социальный блок должен быть отражён.

Если есть учёба или проектный эпизод хотя бы в одном дне, он должен быть отражён хотя бы локально.

Если один `main_events` item требует больше 4 evidence objects, разбей его на несколько более конкретных объектов.

---

### `main_wins`

Агрегируй не только из daily `wins`, но и из:

- `important_moments`;
- `decisions`;
- `short_summary`;
- `key_quotes`.

Win — это:

- завершённое дело;
- преодоление сопротивления;
- спорт / рутина / учёба, если это реально сделано;
- помощь человеку;
- практическое действие, которое автор сделал несмотря на сложность.

Не включай в weekly win:

- игру;
- контент;
- отдых;
- залипание;

если рядом есть evidence, что это было от скуки, из пустоты, как избегание или “не самое лучшее решение”.

---

### `main_tensions`

Агрегируй:

- скуку;
- пустоту;
- вялость;
- усталость;
- сомнения;
- страх ошибиться;
- неприятную работу;
- долгую дорогу;
- внутреннее сопротивление;
- уход от целей;
- open loops;
- сомнительные решения;
- финансовые / бытовые неопределённости.

Если tension встречается в 2+ днях, это repeated weekly tension.

Если tension сильный, он может попасть в `main_tensions` даже из 1 дня.

---

### `emotional_background`

Не смешивай все эмоции в один общий claim.

Если есть разные эмоциональные кластеры, создай несколько объектов.

Примеры кластеров:

- скука / пустота / вялость;
- усталость / плохое настроение;
- интерес / удовольствие / вовлечённость;
- радость от общения;
- тревога / страх / неопределённость;
- облегчение / разрядка.

Каждый кластер должен иметь evidence.

Если evidence взята из daily `emotions`, в weekly evidence пиши:

```json
"source_field": "emotional_signals"
```

Не пиши:

```json
"source_field": "emotions"
```

Не делай глубоких психологических выводов.

---

### `body_energy`

Агрегируй:

- сон;
- усталость;
- вялость;
- бодрость;
- физическую активность;
- спорт;
- прогулки;
- восстановление;
- самочувствие.

Если evidence взята из daily `energy_signals`, `health_signals` или `body_energy_signals`, в weekly evidence пиши:

```json
"source_field": "body_energy_signals"
```

Не пиши:

```json
"source_field": "energy_signals"
```

или:

```json
"source_field": "health_signals"
```

Не добавляй причины, если они не указаны в evidence.

Плохо:

```text
Усталость была из-за недостатка восстановления.
```

если evidence говорит только:

```text
устал
```

Хорошо:

```text
В несколько дней были вялость или усталость, при этом присутствовали спорт и прогулки.
```

---

### `study_and_projects`

Это поле не должно быть пустым, если хотя бы в одном daily JSON есть:

- непустой `study_signals`;
- непустой `ml_ds_signals`;
- important_moment про лабы, курс, проект, RAG, ML, DS, программирование, портфолио;
- обсуждение проекта или профессиональной задачи, если оно находится в daily `important_moments`, `study_signals` или `ml_ds_signals`.

Важно: для evidence в `study_and_projects` НЕ используй `source_field: "professional/project_discussion"`.

Выбирай только canonical source_field по исходному daily-полю:

- если цитата взята из `important_moments` → `source_field: "important_moments"`;
- если цитата взята из `study_signals` → `source_field: "study_signals"`;
- если цитата взята из `ml_ds_signals` → `source_field: "ml_ds_signals"`.

Если сигнал единичный, формулируй осторожно:

```text
Учёба и проекты присутствовали точечно, но не стали главным фокусом недели.
```

Не превращай единичный учебный эпизод в главный фокус недели.  
Но не теряй его.

---

### `social_context`

Агрегируй реальные взаимодействия с людьми:

- семья;
- друзья;
- одногруппники;
- коллеги;
- помощь;
- совместные активности;
- переписки;
- встречи.

Если social_signals есть в 2+ днях, социальный блок недели должен быть отражён.

Не считай разговор с AI социальным сигналом, если речь не о реальном человеке.

---

### `actual_focus`

`actual_focus` — это не смысл недели и не интерпретация.

Это ответ на вопрос:

```text
На что реально ушло время и внимание недели?
```

Строй `actual_focus` из day-by-day map.

Обычно он должен быть 2–4 объектами:

- работа / дорога;
- спорт / тело;
- учёба / проекты;
- отдых / игры / контент;
- социальная активность;
- быт / деньги;
- внутренние переживания / планирование.

Не формулируй `actual_focus` как:

- “поиск смысла”;
- “личное развитие”;
- “осознанность”;
- “самопознание”;

если это не фактический блок времени.

Если evidence для actual_focus меньше 4 дней, проверь, не потеряны ли блоки из других дней.

---

### `repeated_topics`

Ищи все темы, которые встречаются минимум в 2 днях.

Не ограничивайся одной темой.

Проверь минимум:

- boredom / emptiness / sluggishness;
- games / content / entertainment;
- work / commute / money;
- social activity;
- sport / body;
- study / projects;
- goals / planning / life direction;
- decisions / open loops;
- rest / recovery.

Для каждой repeated topic создай отдельный объект.

Если тема есть только в 1 дне, не называй её repeated topic.

---

### `important_contradictions`

Перед тем как оставить это поле пустым, обязательно сделай contradiction scan.

Ищи противоречия между:

- намерениями и фактическими действиями;
- ценностями и повторяющимся поведением;
- желанием планировать жизнь и уходом в игры / контент;
- потребностью в деньгах и сомнением, стоит ли работа сил;
- желанием отдыха и давлением “найти идеал”;
- страхом ошибиться и отсутствием конкретного выбора;
- желанием перестать что-то делать и повторением этого поведения.

Если есть:

- “боюсь”;
- “не самое лучшее решение”;
- “от скуки”;
- “пустота”;
- “уйти от планирования”;
- “сбежать от реальности”;
- repeated games/content;

то `important_contradictions` должен быть проверен особенно внимательно.

Если contradiction подтверждается evidence — добавь.

Если evidence слабая, формулируй осторожно:

```text
Есть возможное напряжение между...
```

Но не оставляй поле пустым автоматически.

---

### `open_loops`

Один объект = один незакрытый вопрос.

Не склеивай разные open loops в один summary.

Источники open loops:

- daily `open_questions`;
- daily `open_loops`;
- `key_quotes`;
- tensions / problems;
- decisions without closure;
- short_summary, если в нём явно есть unresolved issue.

Если evidence взята из daily `open_questions`, в weekly evidence пиши:

```json
"source_field": "open_loops"
```

Не пиши:

```json
"source_field": "open_questions"
```

Не создавай open loop из любой эмоции.

Примеры разных open loops:

- “Оправдался ли ремонт холодильника?”
- “Стоит ли работа потраченных сил?”
- “Как не уходить от планирования жизни в развлечения?”
- “Какой конкретный фокус выбрать дальше?”

---

### `risks_next_week`

Риск должен следовать из evidence.

Риск — это не совет.

Хорошие риски:

- повторение игры / контента как автоматического ответа на скуку;
- потеря фокуса из-за неопределённости;
- усталость от работы и дороги;
- продолжение открытого вопроса без решения;
- распыление внимания.

Не делай драматичных рисков.

---

### `next_week_focus_candidates`

Это кандидаты фокуса, а не мотивационные советы.

Они должны быть:

- конкретными;
- маленькими;
- проверяемыми через неделю;
- основанными на evidence.

Запрещены формулировки:

- повысить осознанность;
- заняться личным развитием;
- искать смысл;
- улучшить планирование;
- стать продуктивнее;
- больше отдыхать.

Хорошие формулировки:

- “Выбрать 1 конкретный фокус недели вне игр/контента.”
- “Закрыть 1 учебный хвост.”
- “Ограничить игры в дни, когда они появляются именно от скуки.”
- “Решить, какую роль работа играет на следующей неделе: деньги, опыт или временная обязанность.”
- “Запланировать 1 восстановительный блок после рабочего дня с дорогой.”

Дай 1–3 кандидата, не больше.

---

### `what_not_to_do`

Это не список запретов на жизнь.

`what_not_to_do` должно быть в зоне контроля пользователя.

Не формулируй как глобальный отказ от:

- работы;
- учёбы;
- людей;
- отдыха;
- игр;
- обязательств.

Плохо:

```text
Избегать скучной работы.
```

Лучше:

```text
Не делать глобальный вывод о работе по одному-двум скучным дням.
```

Плохо:

```text
Избегать принятия сложных решений.
```

если evidence говорит только “боюсь наступить не туда”.

Лучше:

```text
Не превращать страх ошибиться в бесконечное обдумывание без конкретного следующего шага.
```

Только если evidence это поддерживает.

Хороший вариант:

```text
Не использовать игры и контент как автоматический ответ на скуку, если сам оцениваешь это как сомнительное решение.
```

если это подтверждено evidence.

---

## Проход 4. Evidence construction

Каждый пункт в списках должен быть объектом:

```json
{
  "summary": "...",
  "evidence": [
    {
      "date": "YYYY-MM-DD",
      "source_day_md": "data/processed/days_md/YYYY-MM-DD.md",
      "source_daily_json": "data/processed/days_json/YYYY-MM-DD.json",
      "source_field": "important_moments",
      "quote": "...",
      "note": "..."
    }
  ]
}
```

### Evidence rules

1. Не делай claim без evidence.
2. Evidence должна реально поддерживать summary.
3. Если evidence противоречит summary — измени summary.
4. Не используй evidence как украшение.
5. Не добавляй в summary то, чего нет в evidence.
6. Для сильных weekly claims используй evidence из нескольких дней.
7. Для локального claim достаточно 1 дня, но формулировка должна быть локальной.
8. `quote` должен быть взят из daily JSON или очень близко к нему.
9. `note` должно объяснять связь evidence с claim, но не добавлять новых фактов.
10. `source_daily_json` не должен быть `null`.
11. `source_day_md` не должен быть `null`.
12. Каждый item обычно должен иметь 1–4 evidence objects.
13. Если нужно 5+ evidence objects, разбей item на несколько более конкретных items.

---

## Проход 5. Claim scope check

Масштаб claim должен соответствовать evidence.

Правила:

- evidence из 1 дня → локальный эпизод;
- evidence из 2 дней → repeated signal в нескольких днях;
- evidence из 3+ дней → заметная weekly theme;
- evidence из 4+ дней или большинства included days → можно говорить о week-level focus / essence.

Не используй широкие формулировки, если evidence слабая.

Плохой пример:

```text
Неделя прошла преимущественно в отдыхе и играх.
```

если evidence только из 2 дней.

Хороший пример:

```text
В нескольких днях игры и контент появлялись как способ заполнить скуку или пустоту.
```

---

## Проход 6. Final quality gates

Перед ответом проверь:

1. Каждый included day был рассмотрен.
2. `week_essence` не строится на 1–2 днях.
3. `actual_focus` отражает фактическую занятость недели, а не красивую интерпретацию.
4. Если есть study/project signals — `study_and_projects` не пустой.
5. Если есть 2+ дня social_signals — `social_context` отражает социальный блок.
6. Если есть 2+ дня games/content — `repeated_topics` отражает игры/контент.
7. Если есть 2+ дня скуки/пустоты/вялости — это отражено в `emotional_background` или `repeated_topics`.
8. Если есть fear/avoidance/goal drift + repeated games/content — `important_contradictions` проверен и не должен быть пустым без причины.
9. Если есть open_questions / open_loops — `open_loops` проверен.
10. `next_week_focus_candidates` конкретны и проверяемы.
11. `what_not_to_do` находится в зоне контроля и поддержано evidence.
12. Нет unsupported causality.
13. Нет source_field вне разрешённого списка.
14. Нет `source_daily_json: null`.
15. Нет evidence object без `source_day_md`.
16. Нет evidence object без `source_daily_json`.
17. Нет top-level JSON без `short_summary`.
18. Нет `source_field: "professional/project_discussion"` или других придуманных source_field.
19. `days_included` скопирован из входных данных без сокращения.
20. Нет markdown, комментариев или текста вне JSON.

---

# 4. Финальная Pydantic self-check

Перед ответом проверь каждый evidence object.

Для каждого evidence object верно:

```json
{
  "date": "...",
  "source_day_md": "...",
  "source_daily_json": "...",
  "source_field": "...",
  "quote": "...",
  "note": "..."
}
```

`source_field` входит только в этот список:

```text
important_moments
wins
tensions
emotional_signals
body_energy_signals
study_signals
ml_ds_signals
social_signals
decisions
open_loops
key_quotes
short_summary
```

Если видишь `professional/project_discussion`, `project_discussion`, `work`, `project`, `emotions`, `open_questions`, `health_signals` или любое другое значение — исправь его до canonical source_field до ответа.

---

# 5. Формат JSON

Верни только валидный JSON.

Не используй markdown fences.  
Не добавляй комментарии до или после JSON.  
Не пиши финальный weekly report.  
Не давай советы в свободном тексте.  
Не добавляй новых фактов.  
Не делай диагнозов и глубоких психологических выводов.

Если информации для поля нет — верни пустой список.

Верни JSON строго в такой структуре:

```json
{
  "type": "week",
  "week_id": "YYYY-Www",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "days_included": [],
  "missing_days": [],

  "week_essence": [],
  "main_events": [],
  "main_wins": [],
  "main_tensions": [],
  "emotional_background": [],
  "body_energy": [],
  "study_and_projects": [],
  "social_context": [],
  "actual_focus": [],
  "repeated_topics": [],
  "important_contradictions": [],
  "open_loops": [],
  "risks_next_week": [],
  "next_week_focus_candidates": [],
  "what_not_to_do": [],

  "short_summary": ""
}
```

Подставь значения:

```json
"type": "week"
"week_id": "{week_id}"
"start_date": "{start_date}"
"end_date": "{end_date}"
"days_included": {days_included}
"missing_days": {missing_days}
```

---

## Требования к `short_summary`

`short_summary` — обязательное поле.

Это 2–4 предложения.

Оно должно:

- сжимать уже доказанные weekly fields;
- не добавлять новых claims;
- отражать фактический фокус недели;
- упомянуть главные repeated topics / tensions, если они есть;
- быть нейтральным;
- не звучать как финальный отчёт.

Не делай summary шире evidence.

Если не получается написать хороший summary, всё равно верни поле:

```json
"short_summary": ""
```
