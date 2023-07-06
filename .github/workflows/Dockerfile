FROM texlive/texlive:latest

COPY ./.github/workflows/build_thesis.sh /app/
COPY ../../thesis /app/

WORKDIR /app
ENTRYPOINT [ "./build_thesis.sh" ]