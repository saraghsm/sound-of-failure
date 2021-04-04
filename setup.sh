mkdir -p ~/.streamlit_run/

echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit_run/config.toml
