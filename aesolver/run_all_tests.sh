#!/bin/bash

for i in "$@"; do
	case $i in
	-q | --quiet_tests)
		quiet_run='true'
		shift
		;;
	-w | --warnings_tests)
		warnings='true'
		shift
		;;
	-v | --verbose)
		verbose_run='true'
		shift
		;;
	*) ;;
	esac
	shift
done

if [[ -z "$quiet_run" ]]; then
	# loud
	quiet_string=''
else
	# quiet
	quiet_string='-q'
fi

if [[ -z "${warnings}" ]]; then
	# warnings
	warning_string='--disable-warnings'
else
	# no warnings
	warning_string=''
fi

if [[ -z "${verbose_run}" ]]; then
	# not verbose
	verbose_string=''
else
	# verbose
	verbose_string='-v'
fi

# get path to this directory
SCRIPTPATH=$(dirname "$0")

# run all tests except the problematic ones
pytest $quiet_string $warning_string $verbose_string --ignore="$SCRIPTPATH"/tests/model_handling/

# now run the problematic ones
pytest $quiet_string $warning_string $verbose_string "$SCRIPTPATH"/tests/model_handling/model_loader/
pytest $quiet_string $warning_string $verbose_string "$SCRIPTPATH"/tests/model_handling/model_reader/
