project('clstm' , ['c'], version: '1.0.0', default_options: ['warning_level=3', 'werror=true'])

subdir('src')

# Source code style guide
cpplint_program = find_program('cpplint', required : false)

if cpplint_program.found()
    # - https://github.com/cpplint/cpplint

    cpplint_config = ['--verbose=0',
      '--linelength=130',
      '--filter=-build/include_subdir,-build/header_guard,-legal/copyright,'+
      '-build/include_what_you_use,-readability/casting,-build/include_subdir,'+
      '-build/include_order,-readability/multiline_string',
      '--verbose=5',
      '--recursive',
      join_paths(meson.current_source_dir(), 'src')]
    cpplint_target = run_target('cpplint', command : [cpplint_program, cpplint_config])

endif

