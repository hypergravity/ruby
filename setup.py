from distutils.core import setup


if __name__ == '__main__':
    setup(
        name='ruby',
        version='0.1.2',
        author='Bo Zhang',
        author_email='bzhang@mpia.de',
        # py_modules=['bopy','spec','core'],
        description='For Bayesian spectroscopy.',  # short description
        license='MIT',
        # install_requires=['numpy>=1.7','scipy','matplotlib','nose'],
        url='http://github.com/hypergravity/ruby',
        classifiers=[
            "Development Status :: 6 - Mature",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3.6",
            "Topic :: Scientific/Engineering :: Astronomy",
            "Topic :: Scientific/Engineering :: Physics"],
        package_dir={'ruby': 'ruby'},
        packages=['ruby', ],
        package_data={"ruby": ["ruby/data/*", "ruby/script/*"]},
        include_package_data=True,
        requires=['numpy', 'scipy', 'matplotlib', 'astropy', 'joblib']
    )
