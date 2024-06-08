#include <iostream>
#include "mjpc/planners/rmp/test/rmp_tester.h"
#include "mjpc/planners/rmp/test/rmp_parser.h"

/**
 * Binary used to generate random worlds, do a run and save the trajectory and possibly the world file.
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char* argv[]){
    Parser parser(argc, argv);
    if(!parser.parse()){
        return 1;
    }

    Tester tester(parser.getParameters(), parser.getSettings());
    tester.run();
}

