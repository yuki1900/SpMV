#pragma once
#include "utils.h"
namespace ReadMatrix {
    using std::string;
    using std::stringstream;
    struct Coordinate {
        int x;
        int y;
        FloatType val;
    };

    inline bool coordcmp(const Coordinate v1, const Coordinate v2)
    {
        if (v1.x != v2.x)
        {
            return (v1.x < v2.x);
        }
        else
        {
            return (v1.y < v2.y);
        }
    }

    // ¶ÁÈ¡¾ØÕó
    void readMatrix(string filename, FloatType** val_ptr, int** cols_ptr,
        int** rowDelimiters_ptr, int* n, int* numRows, int* numCols)
    {
        string line;
        string id;
        string object;
        string format;
        string field;
        string symmetry;

        std::ifstream mfs(filename);
        if (!mfs.good())
        {
            std::cerr << "Error: unable to open matrix file " << filename << std::endl;
            exit(1);
        }
        

        int symmetric = 0;
        int pattern = 0;
        int field_complex = 0;

        int nRows;
        int nCols;
        int nElements;
        // read matrix header
        if (getline(mfs, line).eof())
        {
            std::cerr << "Error: file " << filename << " does not store a matrix" << std::endl;
            exit(1);
        }
        stringstream ss;
        ss.clear();
        ss.str(line);
        ss >> id >> object >> format >> field >> symmetry;

        //sscanf(line.c_str(), "%s %s %s %s %s", id, object, format, field, symmetry);

        if (object != "matrix")
        {
            fprintf(stderr, "Error: file %s does not store a matrix\n", filename.c_str());
            exit(1);
        }

        if (format != "coordinate")
        {
            fprintf(stderr, "Error: matrix representation is dense\n");
            exit(1);
        }

        if (field == "pattern")
        {
            pattern = 1;
        }

        if (field == "complex")
        {
            field_complex = 1;
        }

        if (symmetry == "symmetric")
        {
            symmetric = 1;
        }

        while (!getline(mfs, line).eof())
        {
            if (line[0] != '%')
            {
                break;
            }
        }

        ss.clear();
        ss.str(line);
        ss >> nRows >> nCols >> nElements;
        //sscanf(line.c_str(), "%d %d %d", &nRows, &nCols, &nElements);
        int nElements_padding = (nElements % 16 == 0) ? nElements : (nElements + 16) / 16 * 16;
        //int valSize = nElements_padding * sizeof(struct Coordinate);
        int valSize = nElements_padding;
        if (symmetric)
        {
            valSize *= 2;
        }
        std::vector<Coordinate> coords(valSize);
        //coords = (struct Coordinate*)malloc(valSize);
        int index = 0;
        double xx99 = 0;
        while (!getline(mfs, line).eof())
        {
            ss.clear();
            ss.str(line);
            if (pattern)
            {
                ss >> coords[index].x >> coords[index].y;
                coords[index].val = index % 13;
            }
            else if (field_complex)
            {
                ss >> coords[index].x >> coords[index].y >> coords[index].val >> xx99;
            }
            else
            {
                ss >> coords[index].x >> coords[index].y >> coords[index].val;
            }

            // convert into index-0-as-start representation
            coords[index].x--;
            coords[index].y--;    
            index++;
            if (symmetric && coords[index - 1].x != coords[index - 1].y)
            {
                coords[index].x = coords[index - 1].y;
                coords[index].y = coords[index - 1].x;
                coords[index].val = coords[index - 1].val;
                index++;
            }

        }
        nElements = index;
        nElements_padding = (nElements % 16 == 0) ? nElements : (nElements + 16) / 16 * 16;

        for (int qq = index; qq < nElements_padding; qq++)
        {
            coords[qq].x = coords[index - 1].x;
            coords[qq].y = coords[index - 1].y;
            coords[qq].val = 0;
        }

        //sort the elements
        std::sort(coords.begin(), coords.end(), coordcmp);

        // create CSR data structures
        *n = nElements_padding;
        *numRows = nRows;
        *numCols = nCols;
        *val_ptr = new FloatType[nElements_padding];
        *cols_ptr = new int[nElements_padding];
        *rowDelimiters_ptr = new int[nRows + 2];

        FloatType* val = *val_ptr;
        int* cols = *cols_ptr;
        int* rowDelimiters = *rowDelimiters_ptr;

        rowDelimiters[0] = 0;
        int r = 0;
        int i = 0;
        for (i = 0; i < nElements_padding; i++)
        {
            while (coords[i].x != r)
            {
                rowDelimiters[++r] = i;
            }
            val[i] = coords[i].val;
            cols[i] = coords[i].y;
        }

        for (int k = r + 1; k <= (nRows + 1); k++)
        {
            rowDelimiters[k] = i - 1;
        }
    }
}
