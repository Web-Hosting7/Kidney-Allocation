# Organ Allocation Preference Study — Flask edition

## Run
    pip install -r requirements.txt
    flask --app app run --debug
Then open http://127.0.0.1:5000

## Flow implemented
1. `/`                     — start screen (name entry, feature descriptions)
2. `/questionnaire/1`      — 20 randomly generated pairwise scenarios (dependents, age,
                              years_waiting, urgency_score, health_score)
3. `/results`              — trained FFT visualization (fft_svg_explained)
4. `/edit`                 — interactive tree editor (ported from index.html, same
                              SVG/JS, now talks to Flask via fetch() instead of the
                              Streamlit component protocol)
5. `/questionnaire/2`      — 10 more scenarios, then `/final` shows the retrained model

## Notes
- `fft_model.py` and `fft_component.py` are unchanged from the Streamlit app — both were
  already pure/framework-agnostic, so they ported over with zero edits.
- `prior_transplants` is dropped; this cut only uses 5 features per the current spec.
- State/persistence still uses `users.json` + per-user response CSVs, same as before,
  just moved behind Flask's session (signed cookie storing the username) instead of
  st.session_state.
- LLM explanation calls (`explain_node_llm`, etc.) are still wired in fft_model.py and
  will work as before if GROQ credentials are set; not otherwise touched.
- No CSS framework — styling lives in templates/base.html as plain CSS custom properties,
  mirroring the COLORS palette from the old app.py, so you have full control now instead
  of fighting Streamlit's widget styles.
