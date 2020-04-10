""" All config for translating client and server responses """

SERVICE_CONFIG = {
    'API_ENDPOINT': 'http://localhost:8501/v1/models/my_baseline:predict',
}

CONTRACT_CONFIG = {
    'CLASSES': ['albrecht-durer', 'alfred-sisley', 'amedeo-modigliani', 'boris-kustodiev',
                'camille-corot', 'camille-pissarro', 'childe-hassam', 'claude-monet',
                'david-burliuk', 'edgar-degas', 'ernst-ludwig-kirchner', 'eugene-boudin',
                'francisco-goya', 'gustave-dore', 'henri-de-toulouse-lautrec', 'henri-matisse',
                'ilya-repin', 'isaac-levitan', 'ivan-aivazovsky', 'ivan-shishkin', 'james-tissot',
                'joaquã\xadn-sorolla', 'john-singer-sargent', 'konstantin-korovin',
                'konstantin-makovsky', 'marc-chagall', 'martiros-saryan', 'maurice-prendergast',
                'nicholas-roerich', 'odilon-redon', 'pablo-picasso', 'paul-cezanne', 'paul-gauguin',
                'peter-paul-rubens', 'pierre-auguste-renoir', 'pyotr-konchalovsky',
                'raphael-kirchner', 'rembrandt', 'salvador-dali', 'sam-francis', 'thomas-eakins',
                'utagawa-kuniyoshi', 'vincent-van-gogh', 'william-merritt-chase',
                'zinaida-serebriakova'],
    'IMAGE_SIZE': (224, 224),
    'RESCALE_FACTOR': 1./255
}