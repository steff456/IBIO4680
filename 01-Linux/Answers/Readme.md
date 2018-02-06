#Computer vision 

##Lab 1 Linux - Git

###Sergio Alberto Galindo León, Stephannie Jimenez Gacha

1. `grep` is a Unix command that finds all lines in the text documents of an specified directory or file that contains (embedded or not) a given word, words or phrase [1]. It’s syntax is:

```
grep    [options]    word_phrase_words   directory_or_file 
```

The command is case sensitive and shows the name of the file or files containing the search terms if there is more than one output file. The output of the grep command has the following form:

```
./file1:line_containing_search_term
./file2:line_containing_search_term
.
.
.
```

Returning one line per each coincidence. The command is mainly used for searching files or documents in large directories [1].

2. The  `#!  /bin/bash` at the start of the scripts is called the shebang and it is used in order to allow the text and data files to be used as executable files under the bash shell in Linux. After the `#!` characters the route of the desired interpreter is written. In this way, the statements contained in the file are interpreted with the specific interpreter found in the given directory of the shebang [2]. It’s syntax is:

```
#!route_of_interpreter   [options_or_arguments]
```

3.Using `cat /etc/passwd|wc -l ` in the server terminal it was possible to determine that the number of users in the course servers is 38. We counted the number of lines of the etc/passwd file which contains one line per user [3] [4].

4. `cut -f 1,7 -d: /etc/passwd | tr ":" " " | sort -k2`

The `cut -f 1,7 -d: /etc/passwd` command will trim the contents of the passwd file respecting the first and seventh field  (-f 1,7) defined by the delimiter `“ : ”(-d:)`, after that, this information is passed to the tr command that replaces all `“ : ”` by spaces `“ ”` in order to have a table separated by spaces, then, this new table is passed to the sort command that organizes the lines respecting the second field (-k2) whch corresponds to the original seventh field: the shell of each user [5].

5. 