cc = meson.get_compiler('c')

m_dep = cc.find_library('m', required: true)

includes = include_directories('.')
sources = ['layers.c','main.c','set.c','utilities.c', 'lstm.c']

network = executable('net',
  sources: [sources],
  dependencies: m_dep,
  include_directories: [includes],
  install: true
)
