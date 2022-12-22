import builtins, coverage, func_timeout, inspect, logging, multiprocessing, os, pickle, sys, traceback
from libct.constraint import Constraint
from libct.path import PathToConstraint
from libct.solver import Solver
from libct.utils import ConcolicObject, unwrap, get_module_from_rootdir_and_modpath, get_function_from_module_and_funcname
import cProfile
import sys
import time




log = logging.getLogger("ct.explore")
sys.setrecursionlimit(1000000) # The original limit is not enough in some special cases.

def prepare():
    #################################################################
    # Since the source code in https://github.com/python/cpython/blob/e822e37946f27c09953bb5733acf3b07c2db690f/Modules/socketmodule.c#L6485
    # only accepts "unwrapped" input arguments, we simply do it here.
    #################################################################
    import socket
    _socket_getaddrinfo = socket.getaddrinfo
    def socket_getaddrinfo(*args, **kwargs):
        return _socket_getaddrinfo(*map(unwrap, args), **{k: unwrap(v) for (k, v) in kwargs.items()})
    socket.getaddrinfo = socket_getaddrinfo
    #####################################################################
    # The builtin len(...) function will automatically unwrap our result,
    # so we want to avoid this by doing the following line.
    #####################################################################
    builtins.len = lambda x: x.__len__()

class ExplorationEngine:
    class Exception(metaclass=type('', (type,), {"__repr__": lambda self: '<EXCEPTION>'})): pass # indicate occurrence of Exception during execution
    class Timeout(metaclass=type('', (type,), {"__repr__": lambda self: '<TIMEOUT>'})): pass # indicate timeout after either a concolic or a primitive execution
    class Unpicklable(metaclass=type('', (type,), {"__repr__": lambda self: '<UNPICKLABLE>'})): pass # indicate that an object is unpicklable
    class LazyLoading(metaclass=type('', (type,), {"__repr__": lambda self: '<DEFAULT>'})): pass # lazily loading default values of primitive arguments

    def __init__(self, *, solver='cvc4', timeout=20, safety=0, store=None, verbose=4, logfile=None, statsdir=None):
        self.__init2__(); self.statsdir = statsdir
        if self.statsdir: os.system(f"rm -rf '{statsdir}'"); os.system(f"mkdir -p '{statsdir}'")
        Solver.set_basic_configurations(solver, timeout, safety, store, statsdir)
        ############################################################
        # This section mainly deals with the logging settings.
        log_level = 25 - 5 * verbose
        if logfile:
            logging.basicConfig(filename=logfile, level=log_level,
                                format='%(asctime)s  %(name)s\t%(levelname)s\t %(message)s',
                                datefmt='%m/%d/%Y %I:%M:%S %p')
        elif logfile == '':
            logging.basicConfig(level=logging.CRITICAL+1)
        else:
            logging.basicConfig(level=log_level,# stream=sys.stdout,
                                format='  %(name)s\t%(levelname)s\t %(message)s')
        ############################################################
        # We add our new logging level called "SMTLIB2" to print out
        # messages related to the solving process.
        ############################################################
        logging.SMTLIB2 = (logging.DEBUG + logging.INFO) // 2
        logging.addLevelName(logging.SMTLIB2, "SMTLIB2")
        def smtlib2(self, message, *args, **kwargs): # https://stackoverflow.com/questions/2183233/how-to-add-a-custom-loglevel-to-pythons-logging-facility/13638084#13638084
            if self.isEnabledFor(logging.SMTLIB2): # Yes, logger takes its '*args' as 'args'.
                self._log(logging.SMTLIB2, message, args, **kwargs)
        logging.Logger.smtlib2 = smtlib2

    def __init2__(self):
        self.constraints_to_solve = [] # consists of the constraints that are going to be solved by the solver
        self.path = PathToConstraint()
        self.in_out = []
        self.coverage_data = coverage.CoverageData()
        self.coverage_accumulated_missing_lines = {}
        self.var_to_types = {}
        self.concolic_name_list = [] ##NOTE for DNN testing
        self.concolic_flag_dict = {} ##NOTE for DNN testing
        self.normalize = False
        # XiaoDai
        self.previous_result = None

    def _execution_loop(self, max_iterations, all_args, concolic_dict):
        start = time.time()
        Solver.normalization = self.normalize
        tried_input_args = [all_args.copy()] # .copy() is important!!
        iterations = 1; cont = self._one_execution(all_args, concolic_dict) # the 1st execution
        old_time = start
        while cont and (max_iterations==0 or iterations<max_iterations) and len(self.constraints_to_solve) > 0:
            ##############################################################
            # In each iteration, we take one constraint out of the queue
            # and try to solve for it. After that we'll obtain a model as
            # a list of arguments and continue the next iteration with it.

            #constraint = self.constraints_to_solve.pop(0) #queue
            #NOTE stack is used instead of queue for DNN
            constraint = self.constraints_to_solve.pop() #stack
            model = Solver.find_model_from_constraint(self, constraint)
            #想存產生的新input
            ##############################################################
            if model is not None:
                all_args.update(model) # from model to argument
                if all_args not in tried_input_args:
                    tried_input_args.append(all_args.copy()) # .copy() is important!!
                    log.info(f"=== Iterations: {iterations} ==="); iterations += 1
                    cont = self._one_execution(all_args, concolic_dict) # other consecutive executions following the 1st execution
                    print('this iteration cost time = ',time.time()-old_time)
                    old_time = time.time()
                            #想存產生的新input
        now = time.time()
        elapsed = now - start
        print('total time = ',elapsed)
        return iterations

    def _can_use_concolic_wrapper(self, root, modpath):
        r, s = multiprocessing.Pipe()
        if os.fork() == 0: # child process
            try:
                import libct.wrapper
                module = get_module_from_rootdir_and_modpath(root, modpath)
                s.send(1)
            except:
                s.send(0)
            os._exit(os.EX_OK)
        os.wait()
        ans = r.recv(); r.close(); s.close()
        return ans

    def explore(self, modpath, all_args={}, /, *, root='.', funcname=None, max_iterations=0, single_timeout=45, total_timeout=2700, deadcode=set(), 
                include_exception=False, lib=None, single_coverage=True, file_as_total=False, concolic_dict={}, norm=False):
        self.modpath = modpath; self.funcname = funcname; self.single_timeout = single_timeout; self.total_timeout = total_timeout; self.include_exception = include_exception; self.deadcode = deadcode; self.lib = lib; self.file_as_total = file_as_total; self.normalize = norm
        if self.funcname is None: self.funcname = self.modpath.split('.')[-1]
        self.__init2__(); self.root = os.path.abspath(root); self.target_file = self.root + '/' + self.modpath.replace('.', '/') + '.py'
        self.single_coverage = single_coverage
        if self.single_coverage:
            self.coverage = coverage.Coverage(data_file=None, include=[self.target_file])
        else:
            self.coverage = coverage.Coverage(data_file=None, source=[self.root], omit=['**/__pycache__/**', '**/.venv/**'])
        if self.lib: sys.path.insert(0, os.path.abspath(self.lib))
        sys.path.insert(0, self.root)#; sys.path.insert(0, file_dir)
        file_dir = os.path.abspath(os.path.dirname(os.path.join(self.root, self.modpath.replace('.', '/') + '.py')))
        now_dir = os.getcwd(); os.chdir(file_dir)
        self.can_use_concolic_wrapper = self._can_use_concolic_wrapper(self.root, self.modpath)
        try: iterations = func_timeout.func_timeout(self.total_timeout, self._execution_loop, args=(max_iterations, all_args, concolic_dict))
        except BaseException as e: # importantly note that func_timeout.FunctionTimedOut is NOT inherited from the (general) Exception class.
            print('Was this exception triggered by total_timeout? ' + str(e))
            iterations = 0 # usually catches timeout exceptions
            traceback.print_exc()
            os._exit(1)
        os.chdir(now_dir); del sys.path[0]
        if self.lib: del sys.path[0]
        if self.statsdir:
            with open(self.statsdir + '/inputs.pkl', 'wb') as f:
                pickle.dump([e[0] for e in self.in_out], f) # store only inputs
            if self.single_coverage:
                with open(self.statsdir + '/missing_lines.txt', 'w') as f:
                    if self.file_as_total:
                        f.write(str(sorted(self.module_lines_range & self.coverage_accumulated_missing_lines[self.target_file])) + '\n')
                        f.write(str(sorted(self.module_lines_range)) + '\n')
                    else:
                        f.write(str(sorted(self.function_lines_range & self.coverage_accumulated_missing_lines[self.target_file])) + '\n')
                        f.write(str(sorted(self.function_lines_range)) + '\n')
            with open(self.statsdir + '/smt.csv', 'w') as f:
                f.write(',number,time\n')
                f.write(f'sat,{Solver.stats["sat_number"]},{Solver.stats["sat_time"]}\n')
                f.write(f'unsat,{Solver.stats["unsat_number"]},{Solver.stats["unsat_time"]}\n')
                f.write(f'otherwise,{Solver.stats["otherwise_number"]},{Solver.stats["otherwise_time"]}\n')
        return (iterations-1, Solver.stats["sat_number"], Solver.stats["sat_time"], Solver.stats["unsat_number"], Solver.stats["unsat_time"], Solver.stats["otherwise_number"], Solver.stats["otherwise_time"])

    def _one_execution(self, all_args, concolic_dict):
        print('#'*80)            
        result = self._one_execution_concolic(all_args, concolic_dict) # primitive input arguments "all_args" may be modified here.
        # answer = self._one_execution_primitive(all_args) # we must measure the coverage in the primitive mode since self.constraints_to_solve would become unpicklable if measured in the concolic mode
        answer = -100
        # result = answer + 1
        # print(f"{'*'*24}  result=@{result}@, answer=@{answer}@  {'*'*24}")
        print(f"{'*'*24}  result=@{result}@, previous_result=@{self.previous_result}@  {'*'*24}")
        
        if self.Timeout not in (result, answer):
            if self.previous_result != None and result != self.previous_result:
                # print('／My result:', result, '／Correct answer:', answer) #'Input:', all_args, 
                print('／My result:', result, '／Previous Result:', self.previous_result) #'Input:', all_args,                 
                return False
                # assert result == answer
            self.previous_result = result

        if not self.single_coverage: # We don't measure coverage in the primitive mode under the non-single coverage setting.
            self.in_out.append((all_args.copy(), result)) # .copy() is important! Think why.
            return True # continue iteration
        # Note only in the self.single_coverage mode does the program go here.
        # if self.file_as_total:
        #     s = (self.module_lines_range - self.deadcode) & self.coverage_accumulated_missing_lines[self.target_file]
        # else:
        #     s = (self.function_lines_range - self.deadcode) & self.coverage_accumulated_missing_lines[self.target_file]
        s = None
        log.info(f"Not Covered Yet: {self.target_file} {sorted(s) if s else '{}'}")
        #FIXME
        return 1
        # return s # continue iteration only if the target file / function coverage is not full yet.

    def _one_execution_concolic(self, all_args, concolic_dict):
        r1, s1 = multiprocessing.Pipe(); r2, s2 = multiprocessing.Pipe(); r3, s3 = multiprocessing.Pipe(); r0, s0 = multiprocessing.Pipe()
        def child_process():
            sys.dont_write_bytecode = True # very important to prevent the later primitive mode from using concolic objects imported here...
            prepare(); self.path.__init__();
            # log.info("Inputs: " + str(all_args))
            if self.can_use_concolic_wrapper:
                import libct.wrapper
            else:
                import libct
            module = get_module_from_rootdir_and_modpath(self.root, self.modpath)
            execute = get_function_from_module_and_funcname(module, self.funcname)
            ccc_args, ccc_kwargs = self._get_concolic_arguments(execute, all_args, concolic_dict) # primitive input arguments "all_args" may be modified here.
            s1.send((all_args, self.var_to_types, self.concolic_name_list, self.concolic_flag_dict)); result = self.Exception
            try:
                result = libct.utils.unwrap(func_timeout.func_timeout(self.single_timeout, execute, args=ccc_args, kwargs=ccc_kwargs))
                print(f"{'*'*20}  result=@{result}@  {'*'*20}")
                log.info(f"Return: {result}")
            except func_timeout.FunctionTimedOut:
                result = self.Timeout
                log.error(f"Timeout (soft) for: {all_args} >> ./pyct.py -r '{self.root}' '{self.modpath}' -s {self.funcname} {{}} --lib '{self.lib}' --include_exception")#; traceback.print_exc()
                if self.statsdir:
                    with open(self.statsdir + '/exception.txt', 'a') as f:
                        print(f"Timeout (soft) for: {all_args} >> ./pyct.py -r '{self.root}' '{self.modpath}' -s {self.funcname} {{}} --lib '{self.lib}' --include_exception", file=f)
            except Exception as e:
                log.error(f"Exception for: {all_args} >> ./pyct '{self.root}' '{self.modpath}' -s {self.funcname} {{}} -m 20 --lib '{self.lib}' --include_exception"); log.error(e); traceback.print_exc()
                if self.statsdir:
                    with open(self.statsdir + '/exception.txt', 'a') as f:
                        print(f"Exception for: {all_args} >> ./pyct '{self.root}' '{self.modpath}' -s {self.funcname} {{}} -m 20 --lib '{self.lib}' --include_exception", file=f); print(e, file=f)
            ###################################### Communication Section ######################################
            s0.send(0) # just a notification to the parent process that we're going to send data
            try: s2.send(result)
            except: s2.send(self.Unpicklable)
            try: s3.send((Constraint.global_constraints, self.constraints_to_solve, self.path))
            except: s3.send(self.Unpicklable) # may fail if they contain some unpicklable objects
        process = multiprocessing.Process(target=child_process); process.start()
        (all_args2, self.var_to_types, self.concolic_name_list, self.concolic_flag_dict) = r1.recv(); r1.close(); s1.close(); all_args.clear(); all_args.update(all_args2) # update the parameter directly
        if not r0.poll(self.single_timeout + 5):
            result = self.Timeout
            log.error(f"Timeout (hard) for: {all_args} >> ./pyct.py -r '{self.root}' '{self.modpath}' -s {self.funcname} {{}} --lib '{self.lib}' --include_exception")
            if self.statsdir:
                with open(self.statsdir + '/exception.txt', 'a') as f:
                    print(f"Timeout (hard) for: {all_args} >> ./pyct.py -r '{self.root}' '{self.modpath}' -s {self.funcname} {{}} --lib '{self.lib}' --include_exception", file=f)
        else:
            result = r2.recv()
            if (t:=r3.recv()) is not self.Unpicklable: (Constraint.global_constraints, self.constraints_to_solve, self.path) = t
        r2.close(); s2.close(); r3.close(); s3.close(); r0.close(); s0.close()
        if process.is_alive(): process.kill()
        return result

    def _one_execution_primitive(self, all_args):
        r1, s1 = multiprocessing.Pipe(); r2, s2 = multiprocessing.Pipe(); r0, s0 = multiprocessing.Pipe()
        def child_process():
            sys.dont_write_bytecode = True # same reason mentioned in the concolic mode
            self.coverage.start()
            module = get_module_from_rootdir_and_modpath(self.root, self.modpath)
            execute = get_function_from_module_and_funcname(module, self.funcname)
            s1.send(set(self.coverage.analysis(self.target_file)[1]) & set(range(1, 1+len(inspect.getsourcelines(module)[0])))) # Note inspect.getsourcelines(module)[1] always returns 0, which is not the fact.
            s1.send(set(self.coverage.analysis(self.target_file)[1]) & set(range(inspect.getsourcelines(execute)[1], inspect.getsourcelines(execute)[1] + len(inspect.getsourcelines(execute)[0]))))
            pri_args, pri_kwargs = self._complete_primitive_arguments(execute, all_args)
            answer = self.Exception
            try:
                answer = func_timeout.func_timeout(self.single_timeout, execute, args=pri_args, kwargs=pri_kwargs)
            except func_timeout.FunctionTimedOut: answer = self.Timeout
            except: pass
            self.coverage.stop(); self.coverage_data.update(self.coverage.get_data())
            for file in self.coverage_data.measured_files(): # "file" is absolute here.
                _, _, missing_lines, _ = self.coverage.analysis(file)
                if file not in self.coverage_accumulated_missing_lines:
                    self.coverage_accumulated_missing_lines[file] = set(missing_lines)
                else:
                    self.coverage_accumulated_missing_lines[file] &= set(missing_lines)
            ###################################### Communication Section ######################################
            s0.send(0) # just a notification to the parent process that we're going to send data
            try: s1.send(answer)
            except: answer = self.Unpicklable; s1.send(answer)
            if self.include_exception or (answer is not self.Exception):
                s2.send((self.coverage_data, self.coverage_accumulated_missing_lines))
            else:
                s2.send(self.Exception)
        process = multiprocessing.Process(target=child_process); process.start()
        self.module_lines_range = r1.recv(); self.function_lines_range = r1.recv()
        if not r0.poll(self.single_timeout + 5): answer = self.Timeout
        else:
            answer = r1.recv()
            if (t:=r2.recv()) is not self.Exception: (self.coverage_data, self.coverage_accumulated_missing_lines) = t
        if self.target_file not in self.coverage_accumulated_missing_lines: self.coverage_accumulated_missing_lines[self.target_file] = self.module_lines_range
        self.in_out.append((all_args.copy(), answer)); r1.close(); s1.close(); r2.close(); s2.close(); r0.close(); s0.close()
        if process.is_alive(): process.kill()
        return answer

    @classmethod
    def _complete_primitive_arguments(cls, func, all_args):
        prim_args = []; prim_kwargs = {}
        for v in inspect.signature(func).parameters.values():
            if v.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD): continue # ignore *args, **kwargs at last
            value = v.default if (t:=all_args[v.name]) is cls.LazyLoading else t
            if v.kind is inspect.Parameter.KEYWORD_ONLY:
                prim_kwargs[v.name] = value
            else: # v.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                prim_args.append(value)
        return prim_args, prim_kwargs

    def _get_concolic_arguments(self, func, prim_args, concolic_dict):
        ccc_args = []; ccc_kwargs = {}
        self.concolic_name_list = []
        for v in inspect.signature(func).parameters.values():
            if v.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                prim_args.pop(v.name, None); continue # do not support *args, **kwargs currently
            if v.name in prim_args:
                value = prim_args[v.name]
            else:
                has_value = False
                if (t:=v.annotation) is not inspect._empty:
                    try: value = t(); has_value = True # may raise TypeError: Cannot instantiate ...
                    except: pass
                if not has_value:
                    if (t:=v.default) is not inspect._empty: value = unwrap(t) # default values may also be wrapped
                    else: value = ''
                prim_args[v.name] = value if type(value) in (bool, float, int, str) else self.LazyLoading

            self.concolic_flag_dict[v.name+'_VAR'] = 0
            if type(value) in (bool, float, int, str) and concolic_dict.get(v.name, 1): 
                #print(v.name + " set to ConcolicObj")
                value = ConcolicObject(value, v.name + '_VAR', self) # '_VAR' is used to avoid name collision
                self.concolic_name_list.append(v.name + '_VAR')
                self.concolic_flag_dict[v.name+'_VAR'] = 1
            if v.kind is inspect.Parameter.KEYWORD_ONLY:
                ccc_kwargs[v.name] = value
            else: # v.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                ccc_args.append(value)
        if not self.var_to_types: # remain unchanged once determined
            for (k, v) in prim_args.items():
                k += '_VAR' # '_VAR' is used to avoid name collision
                if type(v) is bool: self.var_to_types[k] = 'Bool'
                elif type(v) is float: self.var_to_types[k] = 'Real'
                elif type(v) is int: self.var_to_types[k] = 'Int'
                elif type(v) is str: self.var_to_types[k] = 'String'
                else: pass # for some default values that cannot be concolic-ized
        return ccc_args, ccc_kwargs

    def coverage_statistics(self):
        total_lines = 0
        executed_lines = 0
        missing_lines = {}
        for file in self.coverage_data.measured_files():
            executable_lines = set(self.coverage.analysis(file)[1])
            if file == self.target_file and not self.file_as_total:
                executable_lines &= self.function_lines_range
            m_lines = self.coverage_accumulated_missing_lines[file]
            if file == self.target_file and not self.file_as_total:
                m_lines &= self.function_lines_range
            total_lines += len(set(executable_lines))
            executed_lines += len(set(executable_lines)) - len(m_lines) # Do not use "len(set(self.coverage_data.lines(file)))" here!!!
            if m_lines: missing_lines[file] = m_lines
            # print(file, executed_lines, total_lines)
        if self.statsdir:
            with open(self.statsdir + '/coverage.txt', 'w') as f:
                f.write("{}/{} ({:.2%})\n".format(executed_lines, total_lines, (executed_lines/total_lines) if total_lines > 0 else 0))
        return total_lines, executed_lines, missing_lines

    def print_coverage(self):
        total_lines, executed_lines, missing_lines = self.coverage_statistics()
        print("\nLine coverage {}/{} ({:.2%})".format(executed_lines, total_lines, (executed_lines/total_lines) if total_lines > 0 else 0))
        if missing_lines and self.single_coverage:
            print("Missing lines:") # only print this info when the scope of coverage is a single file.
            for file, lines in missing_lines.items():
                print(f"  {file}: {sorted(lines)}")
        print("")
