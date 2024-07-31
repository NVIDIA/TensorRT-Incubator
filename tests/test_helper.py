from tests import helper


def format_contents(title, contents, lang):
    return f"{title}\n{contents}"


class TestProcessCodeBlock:
    def test_non_tripy_types_not_printed_as_locals(self):
        # Non-tripy types should never be shown even if there is no `# doc: no-print-locals`
        block = """
        a = 5
        b = "42"
        """

        _, local_var_lines, _, _ = helper.process_code_block_for_outputs_and_locals(block, block, format_contents)

        assert not local_var_lines

    def test_no_print_locals(self):
        block = """
        # doc: no-print-locals
        gpu = tp.device("gpu")
        cpu = tp.device("cpu")
        """

        _, local_var_lines, _, _ = helper.process_code_block_for_outputs_and_locals(block, block, format_contents)

        assert not local_var_lines
