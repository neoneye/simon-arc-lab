# fmt_grid and task_show is by Mikel Bober-Irizar, under the Apache-2.0 License
# https://github.com/mxbi/arckit
from .task import Task
from rich import print as rich_print
from rich.table import Table
from rich.text import Text

def idx2chr(idx):
    return chr(idx + 65)

def fmt_grid(grid, colour=True, spaces=True):
    grid_str = []
    if not colour:
        for row in grid:
            if spaces == 'gpt':
                grid_str.append(''.join([' ' + str(x) for x in row]))
            elif spaces:
                grid_str.append(' '.join([str(x) for x in row]))
            else:
                grid_str.append(''.join([str(x) for x in row]))

        return "\n".join(grid_str)
    else:
        if spaces:
            cmap = dict({i: (str(i) + ' ', f"color({i})") for i in range(10)}, **{str(i): (str(i) + ' ', f"color({i})") for i in range(10)})
        else:
            cmap = dict({i: (str(i), f"color({i})") for i in range(10)}, **{str(i): (str(i), f"color({i})") for i in range(10)})

        for row in grid:
            grid_str += [cmap[digit] for digit in row]
            grid_str += ["\n"]

        return Text.assemble(*grid_str[:-1])

def task_show(task: Task, answer=True):
    table = Table(title=repr(task), show_lines=True)

    data = []
    for i in range(task.count_examples):
        input = task.input_images[i]
        output = task.output_images[i]
        data += [fmt_grid(input), fmt_grid(output)]
        ix, iy = input.shape
        ox, oy = output.shape
        table.add_column(f"{idx2chr(i)}-in {ix}x{iy}", justify="center", no_wrap=True)
        table.add_column(f"{idx2chr(i)}-out {ox}x{oy}", justify="center", no_wrap=True)

    data.append('')
    table.add_column("")
    for i in range(task.count_tests):
        input = task.input_images[i + task.count_examples]
        output = task.output_images[i + task.count_examples]
        table.add_column(f"T{idx2chr(i)}-in", justify="center", header_style="bold", no_wrap=True)
        if answer:
            table.add_column(f"T{idx2chr(i)}-out", justify="center", header_style="bold", no_wrap=True)
            data += [fmt_grid(input), fmt_grid(output)]
        else:
            data += [fmt_grid(input)]

    table.add_row(*data)
    rich_print(table)
    return table
