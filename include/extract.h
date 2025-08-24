#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <errno.h>

int fileExists(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (f) {
        fclose(f);
        return 1;
    }
    return 0;
}

int extract(const char *path) {
    char cmd[512];
    int status;
    status = system("rm -f video_data/*");
    snprintf(cmd, sizeof(cmd),
             "ffmpeg -y   -hwaccel cuda -hwaccel_output_format cuda   -i \"%s\"   -vf \"hwdownload,format=nv12,format=yuvj420p\"   -fps_mode passthrough -q:v 2   video_data/img%%04d.jpg",
             path);
    status = system(cmd);

    return 0;
}

int countFiles(const char *path) {
    DIR *dir;
    struct dirent *entry;
    int count = 0;

    dir = opendir(path);
    if (dir == NULL) {
        perror("opendir failed");
        return -1;
    }

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.' &&
            (entry->d_name[1] == '\0' || 
             (entry->d_name[1] == '.' && entry->d_name[2] == '\0'))) {
            continue;
        }
        count++;
    }

    closedir(dir);
    return count;
}
