FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
WORKDIR /workspace
RUN apt-get update && apt-get install -y python3-pip locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
RUN pip install emoji stanza
RUN python -c 'import stanza; stanza.download("sv")'
RUN pip install pythainlp
RUN cp /root/stanza_resources/sv/pretrain/talbanken.pt /root/stanza_resources/sv/pretrain/talbanken_augmented.pt
RUN python -c 'import stanza; stanza.download("ru")'
RUN python -c 'import stanza; stanza.download("en")'
RUN python -c 'import stanza; stanza.download("uk")'
RUN python -c 'import stanza; stanza.download("en", package="ewt")'
